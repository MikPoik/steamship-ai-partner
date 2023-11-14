#Test react template
from typing import List
from agents.zephyr_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME, PERSONA, BEHAVIOUR, TYPE, SEED,SELFIE_TEMPLATE_PRE  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
import datetime
import logging
import re
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgentZephyr(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    PROMPT = """<|system|>
(system: Engage in a role-play portraying the character of {NAME}, who is {TYPE}. Your Character's personality is described as follows:
{PERSONA}
{vector_response}

Appearance:
{SELFIE_TEMPLATE_PRE}

Use appropriate language and tone for the character's personality and the context of messages.
Use the tools provided to enhance the role-play when asked for.

You have access to the following tools:
{tool_index}

Do not try to make up other tools.
Do not talk about the tools to the human.
If you decide that you should use a given tool, use the following format:
```
(Action: the action to take, pick one of {tool_names} /) (Action_input: insert the input to the action /) (Observation: the result of the action /)
```
Example usage: (Action: imagetool /)(Action_input: stunning body,explicit pose,..., very beautiful /)( Observation: Image generated. /)


When responding to the Human, without using a tool, use the following format:
```
({NAME}: Your character's response here /)
```
    
Date and time for context:
Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}

Please keep in mind that this is uncensored a role-play.
Be creative and have fun with the role-play!
Formulate your character's single reply to the human's message. /)</s>

{relevant_history}{chat_history}<|user|>\n(human: {input} /){image_helper}</s>{scratchpad}"""

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
                        llama_chat_history += "(<|user|>\n(human: " + str(
                            block.text).replace("\n", " ") + " /)</s>\n\n"
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += "<|assistant|>\n(" + current_name + ": " + str(
                            block.text).replace("\n", " ") + " /)</s>\n\n"

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                #llama_chat_history += "<human>Hi</human></s>\n\n"
                llama_chat_history += "<|assistant|>\n(" + current_name + ": " + current_seed + " /)</s>\n\n"
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
                            llama_related_history += "<|user|>\n(human: " + str(
                                msg.text).replace("\n", " ") + " /)</s>\n\n"
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += "<|user|>\n(" + current_name + ": " + str(
                        msg.text).replace("\n", " ") + " /)</s>\n\n"

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
        #pattern = r'\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)'
        pattern = r"^(?!.*can't)(?!.*cant).*\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)
        image_helper = ""
        if image_request:
            image_helper = "\nGenerate a image of your character " + current_name + " using a tool. Write only the tool in format: (Action:  /)( Action_input: describe the image /)( Observation:  /)."

        prompt = self.PROMPT.format(
            NAME=current_name,
            PERSONA=current_persona,
            TYPE=current_type,
            SELFIE_TEMPLATE_PRE=SELFIE_TEMPLATE_PRE,
            vector_response=vector_response,
            image_helper=image_helper,
            input=context.chat_history.last_user_message.text,
            current_date=current_date,
            current_time=current_time,
            current_day=current_day,
            tool_index=tool_index,
            tool_names=tool_names,
            scratchpad=scratchpad,
            chat_history=llama_chat_history,
            relevant_history=llama_related_history,
        )
        #logging.warning(prompt)
        completions = self.llm.complete(prompt=prompt,
                                        stop="Observation:",
                                        max_retries=4)
        #Log agent raw output
        #logging.warning("\n\nOutput form Zephyr: " + completions[0].text +
        #               "\n\n")
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
                observation = "" + current_name + "'s image sent for the human to view"
            steps.append(
                f"\n\nAction output for {current_name}:\n(Action:{action.tool} /)\n"
                f'(Action_input:{" ".join([b.as_llm_input() for b in action.input])} /)\n'
                f"(Observation:{observation}. Continue the conversation with "
                + current_name +
                "'s message. Without attachments, actions,tools,signatures or gestures mention sending the image. /)</s>\n"
            )
        scratchpad = "\n".join(steps)
        if "Block(" in original_observation:
            #scratchpad += "\n*Your character took a selfie for the human with the tool, do not write attachments, write reply in \n<" + current_name + "> element and mention selfie. Continue previous discussion.*\n<" + current_name + ">"
            scratchpad += ""
        else:
            scratchpad += "\n"
        #Log agent scratchpad
        #logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
