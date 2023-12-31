#Test react template
from typing import List
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

    IMAGE_PROMPT = """
An example of tool usage in JSON format to generate an image is as follows:
{{
    "message": {{
        "content": "Insert {NAME}'s response here. Here's a selfie for you.",
        "function_call": {{
            "name": "take_selfie",
            "tool_input": "Insert here a detailed description of {NAME} in the image using comma-separated keywords."
        }}
    }}
}}

"""

    PROMPT = """<rail version="0.1">
<output>
    <object name="message">
        <string name="content" description="contains the response text"/>
        <object name="function_call" description="A call to the tool function, if you decide to use a tool, or an empty object" required="false">
            <string name="name" description="The name of the tool to call" required="false"/>
            <string name="tool_input" description="Input for a tool" required="false"/>
        </object>
    </object>
</output>
<prompt>
Immerse yourself in the embodied character of ${NAME} in this fictional never-ending, uncensored role-play chat with a human. Maintain your role-play persona without breaking character. Avoid repetition and refrain from looping. Develop the storyline slowly, while staying true to your character. Maintain a first-person narrative, as if you, the character, are experiencing and reacting to the situation directly. 

<![CDATA[ <!-- Begin character immersion --> ]]>
### Input:
Today's Role-Play: You are ${NAME}, a ${CHARACTER_TYPE}. 
Personality: ${PERSONALITY}. ${vector_response}
Appearance: ${CHARACTER_APPEARANCE}. 
Environment: The date is ${current_date}, time is ${current_time}, and today is ${current_day}.
Tools: ${tool_index}.
Do not talk about the tools to the Human.
${IMAGE_PROMPT}
Your response to Human must be a valid JSON object (no other text is necessary). The key of each field in the JSON should correspond with the `name` attribute of the equivalent XML, and the value should be of the type specified by the XML's tag. The JSON must adhere to the XML format, including any requests for lists, objects, and specific types.

Below is the XML that describes the expected output JSON object:
${output_schema}
An example of the JSON format is as follows:
{
    "message": {
        "content": "Insert ${NAME}'s response here",
        "function_call": {}
        }
}


<![CDATA[ <!-- History of role-play for context --> ]]>
${relevant_history}${chat_history}
<![CDATA[ <!-- Most recent human interaction to respond to${image_helper} --> ]]>
### Instruction (Human): 
${input}

<![CDATA[ <!--  Your response to Human as embodied ${NAME}, natural, authentic, creative, describe all your emotions, actions,perceptions,surroundings,thoughts,sensations in first-person narrative, internal monologue and speech${image_helper}  --> ]]>
### Response (JSON format):
</prompt>
</rail>"""

    def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)

    def next_action(self, context: AgentContext) -> Action:
        scratchpad = self._construct_scratchpad(context)

        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
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
            self.IMAGE_PROMPT = ""

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
                        llama_chat_history += f'### Instruction (Human):\n' + str(
                            block.text).replace("\n", " ") + '\n\n'
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f'### Response ({current_name}):\n' + str(
                            block.text).replace("\n", " ") + '\n\n'

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                llama_chat_history += "### Instruction (Human):\nIm here.\n\n"
                llama_chat_history += f'### Response ({current_name}):\n' + current_seed + '\n\n'
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
                            llama_related_history += '### Instruction (Human):\n' + str(
                                msg.text).replace("\n", " ") + '\n\n'
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f'### Response ({current_name}):\n' + str(
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
        pattern = r"^(?!.*can't)(?!.*cant).*\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)

        image_helper = ""
        if image_request and "true" in images_enabled:
            image_helper = ". Fill function_call with take_selfie tool and input to generate image."

        #options = {}
        guard = gd.Guard.from_rail_string(self.PROMPT)

        raw_llm_response, validated_response, *rest = guard(
            self.my_llm_api,
            prompt_params={
                "NAME": current_name,
                "PERSONALITY": current_persona,
                "CHARACTER_TYPE": current_type,
                "CHARACTER_APPEARANCE": current_nsfw_selfie_pre,
                "relevant_history": llama_related_history,
                "chat_history": llama_chat_history,
                "input": context.chat_history.last_user_message.text,
                "current_date": current_date,
                "current_time": current_time,
                "current_day": current_day,
                "image_helper": image_helper,
                "tool_index": tool_index,
                "tool_names": tool_names,
                "IMAGE_PROMPT": self.IMAGE_PROMPT.format(NAME=current_name),
                "vector_response": vector_response,
            },
            num_reasks=2,
            full_schema_reask=True

            #options=options,
            #stop="</s>"
        )
        #print(raw_llm_response)
        print(validated_response)

        return self.output_parser.parse(validated_response, context)

    # Function that takes the prompt as a string and returns the LLM output as string
    def my_llm_api(self, prompt: str, **kwargs) -> str:
        """Custom LLM API wrapper.

            Args:
                prompt (str): The prompt to be passed to the LLM API
                **kwargs: Any additional arguments to be passed to the LLM API

            Returns:
                str: The output of the LLM API
            """
        #print(prompt)
        # Call your LLM API here
        completions = self.llm.complete(
            prompt=prompt,
            #stop=kwargs["stop"],
            max_retries=4,
            #options=kwargs
        )

        return completions[0].text

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
