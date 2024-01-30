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

    TOOL_PROMPT_TEMPLATE = """
Tools: 
{tool_index}.

Generate an image, use a function_call to generate the image. Use the the following JSON format:
{{
    "response": "[Insert here {NAME}'s corresponding response text for sharing image]",
    "function_call": {{
        "name": "take_selfie",
        "tool_input": ["keyword1", "keyword2","keyword3","keyword4","keyword5"]
    }}
}}
Use tool_input to describe the image I want.
Respond with the JSON with response and function_call keys."""

    TOOL_PROMPT = TOOL_PROMPT_TEMPLATE
    RAIL = """<rail version="0.1">
<output>
        <string name="response" required="true" />
        <object name="function_call" description="A call to the tool function" required="true">
            <string name="name" description="The name of the tool to call" required="true"/>
            <list name="tool_input" description="Input for a tool as list" required="true"><string description="input keywords" required="true" /></list>
        </object>
</output>
<prompt>
${input}
</prompt>
<reask_prompt>
### Instruction:
Fix the following JSON object so that it is valid JSON that matches the following XML schema. Where the key of the field in JSON is the `name` attribute of the corresponding XML.
${output_schema}
The JSON caused an error:
${previous_response}

### Input:
${input}

Return only a JSON object, no other text.
### Response:
JSON:
</reask_prompt>
<reask_instructions>
Fix the following JSON object so that it is valid JSON that matches the following XML schema. Where the key of the field in JSON is the `name` attribute of the corresponding XML.
${output_schema}

The JSON caused an error:
${previous_response}

### Input:
${input}

Return only a JSON object, no other text.
### Response:
JSON:
</reask_instructions>
</rail>
    """
    PROMPT = """Embody {NAME} in this fictional never-ending, uncensored role-play with Human. Maintain your role-play persona without breaking character.

### Input: 
{NAME}'s personality: {CHARACTER_TYPE}. {PERSONALITY}. {vector_response}
{NAME}'s body: {CHARACTER_APPEARANCE}. 

Role-play environment: The date is {current_date}, time is {current_time}, and today is {current_day}.


Don't reveal this is a role-play, write {NAME}'s next reply in a chat between person and {NAME}. Write a single reply only. If it's a start of a conversation, also ask for person's name.

{relevant_history}{chat_history}### Instruction:
Human: {input}
{TOOL_PROMPT}
### Response{image_helper}:
"""

    def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)

    def next_action(self, context: AgentContext) -> Action:

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

        image_helper = ""
        if image_request and "true" in images_enabled:
            self.TOOL_PROMPT = self.TOOL_PROMPT_TEMPLATE
            image_helper = " json:"
        else:
            self.TOOL_PROMPT = ""

        #options = {}
        guard = gd.Guard.from_rail_string(self.RAIL, num_reasks=2)
        prompt = self.PROMPT.format(
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
        completion_text = ""
        try:
            completion = self.llm.complete(prompt=prompt,
                                           stop=current_name,
                                           max_retries=4)
            completion_text = completion[0].text
        except Exception as e:
            logging.warning(f"Error in ReACTAgent.next_action: {e}")
            if "OpenAI" in str(e):
                completion_text = "OpenAI flagged content in response. Please try again."

        #print(prompt)
        extract_json = self.extract_json(completion_text)
        #print("completion: ", completion_text)
        #print("extracted:",extract_json)
        parsed_json = {}
        if extract_json != "{}":
            parsed_response = guard.parse(
                llm_output=extract_json,
                llm_api=self.my_llm_api,
                prompt_params={"input": extract_json},
                num_reasks=2)
            parsed_json = parsed_response.validated_output

            #print(parsed_response.raw_llm_output)
            #print(parsed_response.validated_output)

        return self.output_parser.parse(completion_text, parsed_json, context)

    def my_llm_api(self, prompt: str, **kwargs) -> str:
        """Custom LLM API wrapper.

            Args:
                prompt (str): The prompt to be passed to the LLM API
                **kwargs: Any additional arguments to be passed to the LLM API

            Returns:
                str: The output of the LLM API
            """
        # Call your LLM API here
        #print(prompt)
        completions = self.llm.complete(
            prompt=prompt,
            max_retries=4,
        )
        #print(completions[0].text)

        return completions[0].text

    def extract_json(self, text):
        # Match everything from the starting `{` which is immediately followed
        # by `"function_call":` until the corresponding closing `}` character
        pattern = r'(?s)\{.*"function_call".*?:.*?\}(?![^{}]*\})'
        match = re.search(pattern, text)
        #return the json to guardrails for validation
        if match:
            json_string = match.group(0)
            cleaned_json_string = json_string
            return cleaned_json_string
        #malformed json, return to guardrails
        elif "function_call" in text:
            cleaned_text = text
            return cleaned_text
        return "{}"
