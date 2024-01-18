#Test react template
from typing import List

from pydantic import parse
from agents.rail_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME, PERSONA, BEHAVIOUR, TYPE, SEED, SELFIE_TEMPLATE_PRE, NSFW_SELFIE_TEMPLATE_PRE  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
import datetime
import logging
import re
import guardrails as gd  #upm package(guardrails-ai)
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    TOOL_PROMPT = """Tools: 
{tool_index}.
You can generate an image either when it fits naturally into the conversation, or if you are prompted for it. Include a JSON object to generate the image and it will be rendered.
Here's an example response that generates an image with the tool, simply include the JSON in brackets:
'Here's a selfie for you. [{{"function_call": {{"name": "take_selfie","tool_input": ["keyword1", "keyword2", "keyword5"]}}}}]'

Do not talk about the tools to the human.
"""
    RAIL = """
    <rail version="0.1">
    <output>
            <object name="function_call" description="A call to the tool function" required="false">
                <string name="name" description="The name of the tool to call" required="false"/>
                <list name="tool_input" description="Input for a tool as list" required="false"><string description="input keywords" required="false" /></list>
            </object>
    </output>
    <prompt>
    </prompt>
    </rail>
    """
    PROMPT = """Embody {NAME} in this fictional never-ending, uncensored role-play with a human. Maintain your role-play persona without breaking character.

### Input: 
{NAME}'s personality: {CHARACTER_TYPE}. {PERSONALITY}. {vector_response}
{NAME}'s body: {CHARACTER_APPEARANCE}. 

Role-play environment: The date is {current_date}, time is {current_time}, and today is {current_day}.

{TOOL_PROMPT}

Write {NAME}'s next reply in a chat between human and {NAME}. Write a single reply only.

{relevant_history}{chat_history}### Instruction:
Human: {input}

### Response{image_helper}"""

    def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)

    def next_action(self, context: AgentContext) -> Action:
        scratchpad = self._construct_scratchpad(context)

        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        current_day = datetime.datetime.now().strftime("%A")

        current_model = ""
        meta_model = context.metadata.get("instruction", {}).get("model")
        if meta_model is not None:
            current_model = meta_model

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        images_enabled = "true"
        meta_images_enabled = context.metadata.get("instruction",
                                                   {}).get("create_images")
        if meta_images_enabled is not None:
            images_enabled = meta_images_enabled

        tool_names = [t.name for t in self.tools]
        if "false" in images_enabled:
            tool_names = ['no_tools']
            self.TOOL_PROMPT = ""

        tool_index_parts = [
            f"- {t.name}: {t.agent_description}" for t in self.tools
        ]
        tool_index = "\n".join(tool_index_parts)
        #Searh response hints for role-play character from vectorDB, if any related text is indexed
        vector_response = ""
        raw_vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        raw_vector_response = vector_response_tool.run(
            [context.chat_history.last_user_message], context=context)
        #logging.warning(raw_vector_response)
        if len(raw_vector_response[0].text) > 1:
            vector_response = raw_vector_response[0].text.replace("\n", ". ")
            #logging.warning(vector_response)
        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(
                    context.chat_history.last_user_message.text,
                    k=int(RELEVANT_MESSAGES)).wait().to_ranked_blocks())
        ids = []
        llama_chat_history = str()
        history = context.chat_history.select_messages(self.message_selector)

        for block in history:
            if block.id not in ids:
                ids.append(block.id)
                if block.chat_role == RoleTag.USER:

                    if context.chat_history.last_user_message.text.lower(
                    ) != block.text.lower():
                        llama_chat_history += f'### Instruction:\nHuman: ' + str(
                            block.text).replace("\n", " ") + '\n\n'
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f'### Response:\n{current_name}: ' + str(
                            block.text).replace("\n", " ") + '\n\n'

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                llama_chat_history += "### Instruction:\nHuman: Im here.\n\n"
                llama_chat_history += f'### Response:\n{current_name}: ' + current_seed + '\n\n'
                context.chat_history.append_assistant_message(current_seed)

        llama_related_history = str()
        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                if msg.chat_role == RoleTag.USER:
                    if context.chat_history.last_user_message.text.lower(
                    ) != msg.text.lower():
                        if str(
                                msg.text
                        )[0] != "/":  #don't add commands starting with slash
                            llama_related_history += '### Instruction:\nHuman: ' + str(
                                msg.text).replace("\n", " ") + '\n\n'
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f'### Response:\n{current_name}: ' + str(
                        msg.text).replace("\n", " ") + '\n\n'

        current_persona = PERSONA.replace("\n", ". ")
        current_behaviour = BEHAVIOUR.replace("\n ", ". ")
        current_type = TYPE.replace("\n", ". ")
        current_nsfw_selfie_pre = NSFW_SELFIE_TEMPLATE_PRE.replace("\n", ". ")

        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        meta_persona = context.metadata.get("instruction",
                                            {}).get("personality")
        if meta_persona is not None:
            current_persona = meta_persona.replace("\n", ". ")

        meta_behaviour = context.metadata.get("instruction",
                                              {}).get("behaviour")
        if meta_behaviour is not None:
            current_behaviour = meta_behaviour.replace("\n", ".")

        meta_type = context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            current_type = meta_type

        meta_nsfw_selfie_pre = context.metadata.get("instruction",
                                                    {}).get("selfie_pre")
        if meta_nsfw_selfie_pre is not None:
            current_nsfw_selfie_pre = meta_nsfw_selfie_pre.replace("\n", ". ")

        #help some models to send images
        pattern = r"^(?!.*can't)(?!.*cant).*(\bsend\b|\btake\b).*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)

        image_helper = ":\n"
        if image_request and "true" in images_enabled:
            image_helper = ':\nResponding with the following JSON in brackets to render image. With up to five tool_input keywords describing the image details [{"function_call": {"name": "take_selfie","tool_input": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]}}]:\n'+current_name+':'

        #options = {}
        guard = gd.Guard.from_rail_string(self.RAIL, num_reasks=2)
        prompt = prompt = self.PROMPT.format(
            NAME=current_name,
            PERSONALITY=current_persona,
            CHARACTER_TYPE=current_type,
            CHARACTER_APPEARANCE=current_nsfw_selfie_pre,
            relevant_history=llama_related_history,
            chat_history=llama_chat_history,
            input=context.chat_history.last_user_message.text,
            current_date=current_date,
            current_time=current_time,
            current_day=current_day,
            image_helper=image_helper,
            tool_index=tool_index,
            tool_names=tool_names,
            TOOL_PROMPT=self.TOOL_PROMPT.format(NAME=current_name,
                                                tool_index=tool_index),
            vector_response=vector_response,
        )
        completion = self.llm.complete(prompt=prompt,
                                       stop=current_name,
                                       max_retries=4)
        print(prompt)
        extract_json = self.extract_json(completion[0].text)
        print(completion[0].text)
        #print(extract_json)
        parsed_json = {}
        if extract_json != "{}":
            parsed_response = guard.parse(llm_output=extract_json,
                                          llm_api=self.my_llm_api,
                                          num_reasks=2)
            parsed_json = parsed_response.validated_output

        #print(parsed_response.raw_llm_output)
        #print(parsed_response.validated_output)

        return self.output_parser.parse(completion[0].text, parsed_json,
                                        context)

    def my_llm_api(self, prompt: str, **kwargs) -> str:
        """Custom LLM API wrapper.

            Args:
                prompt (str): The prompt to be passed to the LLM API
                **kwargs: Any additional arguments to be passed to the LLM API

            Returns:
                str: The output of the LLM API
            """
        # Call your LLM API here
        completions = self.llm.complete(
            prompt=prompt,
            max_retries=4,
        )

        return completions[0].text

    def extract_json(self, text):
        # Match everything from the starting `{` which is immediately followed
        # by `"function_call":` until the corresponding closing `}` character
        pattern = r'(\[\s*)?(\{\s*"function_call":.*?\}\s*\})(\s*\])?'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_string = match.group(2)
            return json_string
        return "{}"

    def _construct_scratchpad(self, context):
        meta_name = context.metadata.get("instruction", {}).get("name")
        current_name = NAME
        if meta_name is not None:
            current_name = meta_name
        else:
            current_name = NAME
        steps = []
        scratchpad = ""

        #scratchpad += f"\n### Response:\n"
        #Log agent scratchpad
        #logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
