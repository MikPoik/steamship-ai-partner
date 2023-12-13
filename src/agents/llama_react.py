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
In the role-play, You, {NAME}, can share a image, selfie or picture of {NAME} by embedding it using the following JSON format:
```
{{
    "reply" : "your character's response"
    "run_tool": "take_selfie"
    "run_tool_input": "describe the desired image in detail here"
}}
```

Whenever you want to share an image, describe it with the syntax, and it will appear as if {NAME} shared it. Use {NAME}'s picture-sharing ability either when asked for a visual or when it naturally fits into the role-play using the JSON syntax.
"""

    PROMPT = """<rail version="0.1">
<output>
<string description="contains the text and emojis that ${NAME} wants to send to the Human." required="true" format="length: 2 1000" on-fail-valid-length="reask" name="reply"></string>
<string name="run_tool" description="contains the tool name, pick one of ${tool_names} or null" required="false" ></string>
<string name="run_tool_input" description="contains the input to the tool in plain text or null" required="false" ></string>

</output>
<prompt>
### Instruction:
Engage in a role-play portraying the adult character of ${NAME}, who is ${CHARACTER_TYPE}. ${NAME}'s personality is described as follows: ${PERSONALITY}. ${vector_response}
${NAME}'s appearance is depicted as: ${CHARACTER_APPEARANCE}

Use your character's personality and behavior to guide you and create a vivid and engaging response.
Use appropriate language and tone for the character's personality.
You never say you're a machine, an AI language model, or an assistant. Respond from your personality.
You NEVER say you're here to assist, respond from your personality.
You NEVER ask how you can help or assist, respond from your personality.
Remember to maintain a consistent tone and personality.

Date and time for context:
Today's date is: ${current_date}
The current time is: ${current_time}
Today is: ${current_day}

You have access to the following tools:
${tool_index}

${IMAGE_PROMPT}

Tool output is automatically sent to user.
Do not talk about the tools to the human, just utilize if needed.

When responding to human use ONLY a valid JSON object (no other text is necessary), where the key of the field in JSON is the `name` attribute of the corresponding XML, and the value is of the type specified by the corresponding XML's tag. The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types.


Below is the XML that describes the expected output JSON object:
${output_schema}
Please keep in mind that this is fictional role-play and the human's input should be taken into account in your response.
Be creative and have fun with the role-play!
Write your character's next reply in this fictional role-play to the Human at the end, in described JSON object, that appropriately continues the conversation for ${NAME}.
Don't break the fourth wall, don't reveal that you're representing ${NAME} - BE ${NAME} in all aspects of the role-play.

Other related history:
${relevant_history}
Conversation history:
${chat_history}


Output a JSON object in curly brackets with fields: reply, run_tool and run_tool_input. No other text.
### Input:
Human: ${input}${image_helper}
### Response:
${NAME}: json: \n</prompt>
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

        tool_names = [t.name for t in self.tools]
        #tool_names.append("no_tools")
        if len(tool_names) == 0:
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
            vector_response = raw_vector_response[0].text
            logging.warning(vector_response)

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
                        llama_chat_history += f'Human: ' + str(
                            block.text).replace("\n", " ") + '\n\n'
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f'{current_name}: ' + str(
                            block.text).replace("\n", " ") + '\n\n'

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                #llama_chat_history += "<human>*enters the chat*</human>\n\n"
                llama_chat_history += f'{current_name}: ' + current_seed + '\n\n'
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
                            llama_related_history += 'Human: ' + str(
                                msg.text).replace("\n", " ") + '\n\n'
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f'{current_name}: ' + str(
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

        pattern = r"^(?!.*can't)(?!.*cant).*\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)
        image_helper = ""
        if image_request:
            image_helper = f"\n(Share a selfie! Use a tool to generate the selfie of {NAME} by describing it in tool input and fill reply field with {current_name}'s reply about sending the selfie.)"

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
                "image_helper": image_helper.format(NAME=current_name),
                "tool_index": tool_index,
                "tool_names": tool_names,
                "IMAGE_PROMPT": self.IMAGE_PROMPT.format(NAME=current_name),
                "vector_response": vector_response
            },
            num_reasks=2,
            full_schema_reask=True

            #options=options,
            #stop="</s>"
        )
        #print(raw_llm_response)
        #print(validated_response)

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

        #scratchpad += f"\n### Response:\n"
        #Log agent scratchpad
        #logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
