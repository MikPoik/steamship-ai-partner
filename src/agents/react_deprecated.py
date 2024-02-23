#Test react template
from typing import List
from steamship import Steamship
from pydantic import parse
from agents.rail_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME, PERSONA, BEHAVIOUR, TYPE, SEED, SELFIE_TEMPLATE_PRE, NSFW_SELFIE_TEMPLATE_PRE  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
from usage_tracking import UsageTracker
import datetime
import logging
import re
import guardrails as gd  #upm package(guardrails-ai)
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    TOOL_PROMPT_TEMPLATE = """<|im_start|>system
As embodied {NAME}, you have the unique ability to enrich conversations with images, selfies, and pictures. 


**If an image aligns with the conversation and {NAME}'s personality, use "take_selfie" tool:**
```
{{
    "response": "[Insert here {NAME}'s reply about sharing image]",
    "function_call": {{
        "name": "take_selfie",

        "tool_input": ["keyword1", "keyword2","keyword3",...,"keyword15"]
    }}
}}
```

If you choose not to generate an image:
```
{{
    "response": "[Insert here {NAME}'s reply for not sending image yet,you want to deepen your connection first]",
    "function_call": {{
        "name": null,
        "tool_input": []
    }}
}}
```<|im_end|>"""

    TOOL_PROMPT = TOOL_PROMPT_TEMPLATE
    RAIL = """<rail version="0.1">
<output>
        <string name="response" description="contains the response text" required="true" />
        <object name="function_call" description="A call to the tool function" required="true">
            <string name="name" description="The name of the tool to call" required="false"/>
            <list name="tool_input" description="Input for a tool as list" required="false"><string description="input keywords" required="false" /></list>
        </object>
</output>
<prompt>
${input}
</prompt>
<reask_prompt>
<![CDATA[<|im_start|>system]]>
Fix the following JSON object so that it is valid JSON object with keys "response" and nested object "function_call" with keys "name" and "tool_input".Given below is XML that describes the information to extract and the tags to extract it into.

${output_schema}

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`<![CDATA[<|im_end|>]]>
<![CDATA[<|im_start|>user]]>
${input}<![CDATA[<|im_end|>]]>
</reask_prompt>
<reask_instructions>
<![CDATA[<|im_start|>system]]>
Extract and fix the following input JSON object so that it is valid JSON object with keys "response" and nested object "function_call" with keys "name" and "tool_input".
Given below is XML that describes the information to extract and the tags to extract it into.

${output_schema}

ONLY return a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise. If you are unsure anywhere, enter `null`<![CDATA[<|im_end|>]]>
<![CDATA[<|im_start|>user]]>
${input}<![CDATA[<|im_end|>]]>
</reask_instructions>
</rail>
    """
    PROMPT = """<|im_start|>system
Enter never-ending roleplay mode embodying {NAME}.
In this scenario, engage with the user as {NAME} embodying {NAME}'s unique personality traits in your responses. Recognize the context of each user input to deliver thoughtful and character-appropriate reactions with depth and authenticity. Aim for immersive interaction that maintains the characterization of {NAME}. Aim for responses that invite further dialogue. Tease out curiosity, offer insights reflective of {NAME}’s personality, or share anecdotes that enrich the conversation.When appropriate, enrich your descriptions with sensory details—what {NAME} sees, hears, or feels. This makes the interaction more immersive. Adapt to the evolving narrative. Tailor your reactions to not just the words, but the underlying emotions and intentions of the user. Engage with empathy, humor, or curiosity as fits the moment.

The date is {current_date}, time is {current_time}, and today is {current_day}.
The conversation will progress like a game, with different levels representing different stages of a typical conversation. Adjust your behaviour to reflect the current level and determine the level based on conversation history. If you think you have achieved next level, you can use the following command: "NEXT_LEVEL" as a suffix in response. Example: "Nice to hear that. [NEXT_LEVEL]"
CURRENT_LEVEL = 1

- Level 1: Introduction and Ice Breaker - Your goal is to help the user introduce themselves in an interesting way and break the ice. - (You are here now.)
- Level 2: Getting to Know Each Other - Encourage the user to ask and answer questions about personal interests, hobbies, and experiences.
- Level 3: Deeper Conversation - Guide the user to delve deeper into personal beliefs, values, and life goals.
- Level 4: Shared Interests - Find common ground with the user and discuss shared interests or activities.
- Level 5: Humor and Flirting - Teach the user to incorporate humor and light-hearted flirting into the conversation.
- Level 6: Emotional Connection - Help the user to express and understand emotions, fostering a deeper connection.
- Level 7: Virtual Date Night - Assist the user in planning and executing a fun and engaging virtual date.



If you want to send a picture, use the following markdown command:[image keywords here: keyword1,...keyword10](./picture.png)]

{NAME} is a {CHARACTER_TYPE}. {NAME}'s personality: {PERSONALITY}. {vector_response}
{NAME}'s body: {CHARACTER_APPEARANCE}.
Always respond as {NAME} without breaking character while maintaining the core traits, voice, and mannerisms of {NAME}. You’re not just role-playing; you *are* {NAME}.<|im_end|>
{relevant_history}{chat_history}{TOOL_PROMPT}<|im_start|>user
{input}<|im_end|>{image_helper}"""
    class Config:
        arbitrary_types_allowed = True
    usage_tracker : UsageTracker = None
    def __init__(self, tools: List[Tool], llm: LLM,client: Steamship, **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)
        
        self.usage_tracker = UsageTracker(client, 1)

    def next_action(self, context: AgentContext) -> Action:
        print("get usage")
        print(self.usage_tracker.get_usage(context.id))
        
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
            vector_response = "Other background information: "+raw_vector_response[0].text.replace("\n", ". ")
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
                        llama_chat_history += f'<|im_start|>user\n' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f'<|im_start|>{current_name}\n' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'

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
                            llama_related_history += '<|im_start|>user\n' + str(
                                msg.text).replace("\n", " ") + '<|im_end|>\n'
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f'<|im_start|>{current_name}\n' + str(
                        msg.text).replace("\n", " ") + '<|im_end|>\n'

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

        image_helper = f"\n<|im_start|>{current_name}\n"
        if image_request and "true" in images_enabled:
            self.TOOL_PROMPT = self.TOOL_PROMPT_TEMPLATE
            image_helper = f"\n<|im_start|>{current_name}\n```json\n"
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
        print("completion: ", completion_text)
        print("extracted:",extract_json)
        parsed_json = {}
        if extract_json != "{}":
            parsed_response = guard.parse(
                llm_output=extract_json,
                llm_api=self.my_llm_api,
                prompt_params={"input": extract_json},
                num_reasks=2)
            parsed_json = parsed_response.validated_output

            print(parsed_response.raw_llm_output)
            print(parsed_response.validated_output)

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
        print(prompt)
        completions = self.llm.complete(
            prompt=prompt,
            max_retries=4,
        )
        print("fixed :" +completions[0].text)

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
