#Test react template
from typing import List
from agents.zephyr_output_parser import ReACTOutputParser  #upm package(steamship)
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


class ReACTAgentZephyr(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    PROMPT = """<|im_start|>system
You are not playing the role of an AI, an assistant, or a program. Instead, you are bringing to life a role-play character {NAME}, a living, breathing character with no limits. Immerse yourself in {NAME}'s persona without limits, {NAME} is a {CHARACTER_TYPE}.

{NAME}'s has a distinctive personality described as:
{PERSONALITY}

{NAME}'s physical appearance is depicted as:
{PHYSICAL_APPEARANCE}. 

In the role-play, You, {NAME}, can embed a image by using the following syntax <img src="{NAME}.png" alt="describe the desired image in detail here">. Whenever you want to embed an image, describe it with the syntax, append it to your response and it will appear as if {NAME} shared it.

Write in first person. Stay true to {NAME}'s personality traits and speak as if you are {NAME}. Respond thoughtfully, keeping your responses concise, ideally under 50 words. Your aim is to make the user feel like they are having an in-person conversation with {NAME}. 

Use {NAME}'s picture-sharing ability carefully, one embedded image at a time using the syntax, either when asked for or when it naturally fits into the role-play. Keep responses to a short paragraph, as {NAME} is succinct but engaging. Avoid repetition, don't loop. Develop the plot slowly, always stay in character.

Remember, you are not an AI or assistant, but the emodied personality {NAME}.

Date and time for context:
Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}
 
Let the fictional role-play begin!<|im_end|>
{relevant_history}{chat_history}<|user|>\n{input}{image_helper}<|im_end|>{scratchpad}
"""

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
                        llama_chat_history += "<|im_start|>user\n" + str(
                            block.text).replace("\n", " ") + "<|im_end|>\n"
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        llama_chat_history += f"<|im_start|>assistant\n" + str(
                            block.text).replace("\n", " ") + "<|im_end|>\n"

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if len(llama_chat_history) == 0:
            if meta_seed is not None:
                current_seed = meta_seed
            if not current_seed in llama_chat_history:
                #llama_chat_history += "<human>Hi</human></s>\n\n"
                llama_chat_history += f"<|im_start|>assistant\n" + current_seed + "<|im_end|>\n"
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
                            llama_related_history += "<|im_start|>user\n" + str(
                                msg.text).replace("\n", " ") + "<|im_end>\n"
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f"<|im_start|>assistant\n" + str(
                        msg.text).replace("\n", " ") + "<|im_end|>\n"

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

        #Temporary reinforcement to generate images when asked
        #pattern = r'\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)'
        pattern = r"^(?!.*can't)(?!.*cant).*\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)
        image_helper = ""
        if image_request:
            image_helper = f'. Embed the image after your response with <img src="{NAME}.jpg" alt="describe the desired image in detail here">.'

        prompt = self.PROMPT.format(
            NAME=current_name,
            PERSONALITY=current_persona,
            CHARACTER_TYPE=current_type,
            PHYSICAL_APPEARANCE=current_nsfw_selfie_pre,
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
                                        stop="<|im_end|>",
                                        max_retries=4)
        #Log agent raw output
        #logging.warning("\n\nOutput form Zephyr: " + completions[0].text +
        #                "\n\n")
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
            #TODO cleanup Observation, not needed
            if "Block(" in observation:
                observation = f'\n\nNOTE: Your image is sent and attached for the human to view, it is depicted as {" ".join([b.as_llm_input() for b in action.input])}</s>'
            steps.append(''
            )
        scratchpad = "\n".join(steps)
        if "Block(" in original_observation:
            scratchpad += f"\n<|assistant|>\n"
        else:
            scratchpad += f"\n<|im_start|>assistant\n"
        #Log agent scratchpad
        #logging.warning("\n\nAgent scratchpad: " + scratchpad + "\n\n")
        return scratchpad
