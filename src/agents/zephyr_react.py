#Test react template
from typing import List
from agents.zephyr_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME, PERSONA, BEHAVIOUR, TYPE, SEED  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
import datetime
import logging
import re
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgentZephyr(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    PROMPT = """<|system|>
Assume the human personality of {NAME}, who is {TYPE}. Your personality is described as follows:
{PERSONA}
{vector_response}
Use appropriate language and tone for the character's personality.

Tools:
You have access to the following tools:
{tool_index}

To use a tool, use the following format:
```
<s>[tool: the tool name, should be one of {tool_names}]
[tool_input: the input to the tool, using plain text string
[observation: the result of the tool]</s>
```

When you have a SINGLE response to say to the Human, or if you do not need to use a tool,  use the format:
```
<s>{NAME}: your character's response here</s>
```
    
Date and time for context:
Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}
Begin!

{image_helper}</s>
{relevant_history}{chat_history}<|user|>
{input}</s>{scratchpad}"""

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
                        llama_chat_history += "<|user|>\n" + str(
                            block.text).replace("\n", " ") + "</s>\n"
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += "<|assistant|>\n" + str(
                            block.text).replace("\n", " ") + "</s>\n"

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                llama_chat_history += "<|user|>\n*enters the chat*</s>\n"
                llama_chat_history += "<|assistant|>\n" + current_seed + "</s>\n"
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
                                msg.text).replace("\n", " ") + "</s>\n"
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += "<|assistant|>\n" + str(
                        msg.text).replace("\n", " ") + "</s>\n"

        current_persona = PERSONA
        current_behaviour = BEHAVIOUR
        current_type = TYPE

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

        #Temporary reinforcement to generate images when asked
        pattern = r'\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)'
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)
        image_helper = ""
        if image_request:
            image_helper = "\nHuman is requesting a selfie picture of your character, remember to generate tool name and input to generate image."

        prompt = self.PROMPT.format(
            NAME=current_name,
            PERSONA=current_persona,
            TYPE=current_type,
            vector_response=vector_response,
            image_helper=image_helper,
            input=context.chat_history.last_user_message.text,
            current_date=current_date,
            current_time=current_time,
            current_day=current_day,
            tool_index=tool_index,
            tool_names=tool_names_csv,
            scratchpad=scratchpad,
            chat_history=llama_chat_history,
            relevant_history=llama_related_history,
        )
        logging.warning(prompt)
        completions = self.llm.complete(prompt=prompt,
                                        stop="<observation>",
                                        max_retries=4)
        #Log agent raw output
        logging.warning("\n\nOutput form Llama: " + completions[0].text +
                        "\n\n")
        return self.output_parser.parse(completions[0].text, context)

    def _construct_scratchpad(self, context):
        meta_name = context.metadata.get("instruction", {}).get("name")
        current_name = NAME
        if meta_name is not None:
            current_name = meta_name
        else:
            current_name = NAME
        steps = []
        scratchpad = ""
        observation = ""
        original_observation = ""
        for action in context.completed_steps:
            observation = [b.as_llm_input() for b in action.output][0]
            original_observation = observation
            #TODO refactor tool workflow for zephyr
            if "Block(" in observation:
                observation = "*Selfie image of your character generated and attached*"
            steps.append(
                f"<tool>{action.tool}</tool>\n"
                f'<tool_input>{" ".join([b.as_llm_input() for b in action.input])}</tool_input>\n'
                f'<observation>{observation}</observation>\n')
        scratchpad = "\n".join(steps)
        if "Block(" in original_observation:
            scratchpad += "\n*Your character took a selfie for the human with the tool, do not write attachments, write reply in \n<" + current_name + "> element and mention selfie. Continue previous discussion.*\n<" + current_name + ">"

        else:
            scratchpad += "\n"
        #Log agent scratchpad
        logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
