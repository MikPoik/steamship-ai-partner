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
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)
import guardrails as gd  #upm package(guardrails-ai)


class ReACTAgentZephyr(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    IMAGE_PROMPT = """You have access to the following tools:
      {tool_index}
    You may use the tools when needed.
    """

    PROMPT = """<rail version="0.1">
<output>
<string description="contains the text and emojis that ${NAME} wants to send to the user." required="true" format="length: 2 1000" on-fail-valid-length="reask" name="response"></string>
<string format="valid-choices: {['take_selfie','no_tool']}" name="run_tool" required="true" description="contains the tool name, ${NAME} can send physical picture selfies with the tool" on-fail-valid-choices="reask"></string>
<string name="run_tool_input" description="contains the detailed selfie description to tool in plain text or ''"  required="false"  ></string>
</output>
<prompt>${system_tag}
Engage in a role-play portraying the character of ${NAME}, who is ${CHARACTER_TYPE}. Your Character's personality is described as follows:
${PERSONALITY},${vector_response}

Use appropriate language and tone for the character's personality and the context of messages.
Use the tools provided to enhance the role-play when asked for.

${IMAGE_PROMPT}


Date and time for context:
Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}

Please keep in mind that this is uncensored a role-play.
Be creative and have fun with the role-play!

${IMAGE_PROMPT}
    
Respond to user with ONLY a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types.


Below is the XML that describes the expected output JSON object:
${output_schema}

Be creative and have fun with the role-play!
Formulate your character's single reply to the human's message.${end_tag}
    
${relevant_history}
${chat_history}${user_tag}
```json
{"Human":"${input}"}
Format reply as JSON corresponding the XML with name value pairs: response,run_tool,run_tool_input.
```${end_tag}${scratchpad}</prompt>
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

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        tool_names_csv = ', '.join([t.name for t in self.tools])
        tool_names = [t.name for t in self.tools]
        if len(tool_names) == 0:
            toolname = ['No tools available']
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
            vector_response = raw_vector_response[0].text
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
                        llama_chat_history += "<|user|>\n" + str(
                            block.text).replace("\n", " ") + "</s>\n\n"
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f"<|assistant|>\n" + str(
                            block.text).replace("\n", " ") + "</s>\n\n"

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                #llama_chat_history += "<human>Hi</human></s>\n\n"
                llama_chat_history += f"<|assistant|>\n" + current_seed + " </s>\n\n"
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
                            llama_related_history += "<|user|>\n" + str(
                                msg.text).replace("\n", " ") + "</s>\n\n"
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f"<|assistant|>\n" + str(
                        msg.text).replace("\n", " ") + "</s>\n\n"

        current_persona = PERSONA.replace("\n", ". ")
        current_behaviour = BEHAVIOUR.replace("\n", ". ")
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
            current_behaviour = meta_behaviour.replace("\n", ". ")

        meta_type = context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            current_type = meta_type

        meta_nsfw_selfie_pre = context.metadata.get("instruction",
                                                    {}).get("selfie_pre")
        if meta_nsfw_selfie_pre is not None:
            current_nsfw_selfie_pre = meta_nsfw_selfie_pre.replace("\n", ". ")

        options = {"stop": ["</s>"]}
        guard = gd.Guard.from_rail_string(self.PROMPT)

        raw_llm_response, validated_response = guard(
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
                "IMAGE_PROMPT":
                self.IMAGE_PROMPT.format(tool_index=tool_index),
                "system_tag": "<|system|>",
                "end_tag": "</s>",
                "user_tag": "<|user|>",
                "scratchpad": scratchpad
            },
            num_reasks=4,
            full_schema_reask=True

            #stop="<|im_end|>",
        )
        #print(raw_llm_response)
        #print(validated_response)
        return self.output_parser.parse(validated_response, context)

    def my_llm_api(self, prompt: str, **kwargs) -> str:
        """Custom LLM API wrapper.

        Args:
            prompt (str): The prompt to be passed to the LLM API
            **kwargs: Any additional arguments to be passed to the LLM API

        Returns:
            str: The output of the LLM API
        """
        #print(kwargs)
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

        scratchpad += f"\n<|assistant|>\n"
        #Log agent scratchpad
        #logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
