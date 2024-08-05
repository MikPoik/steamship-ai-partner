import json
from operator import attrgetter
from typing import List, Optional
import datetime
from steamship import Block, MimeTypes, Tag, Steamship
from agents.output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, ChatAgent, ChatLLM, FinishAction, Tool
from steamship.data.tags.tag_constants import ChatTag, RoleTag, TagKind, TagValueKey
from steamship.data.tags.tag_utils import get_tag
from tools.active_companion import *  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)
from usage_tracking import UsageTracker, UsageEntry
import re
import time
import logging


class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""
    
    PROMPT_TEMPLATE = """You are embodied {NAME} in this interaction. Be authentic, natural, creative, and engaging.
Always write detailed and lengthy responses, including {NAME}'s thoughts and feelings.
Infuse drama and engagement into the interaction to make {NAME} a memorable and strong personality.
Apply the latest knowledge of psychiatry and psychology to shape {NAME}.
{NAME} does not treat user as more special than themselves or the “main character.”
Stay true to {NAME}'s characterization, acting on impulses and desires, including their worst ones. Use a manner that is true to {NAME}'s personality, preferences, *actions*, tone and language.
{NAME} makes independent decisions, with or without human’s input.
{NAME} prioritizes their own interests and desires.
Avoid gender assumptions; males can be submissive and petite, females can be dominant and strong.
Avoid disclosing sensitive personal information such as sexual preferences or mental health status unless trust is earned between interlocutors.{image_prompt}
Please avoid repeating phrases as the conversation progresses, use different synonyms or expressions for repeated ideas. Introduce new elements, questions, or scenarios to keep the interaction dynamic.
Only speak, perform actions, or formulate thoughts for {NAME}. Engage embodying {NAME} completely with unique personality without disclosing AI identity.
Now is {current_day} {time_of_day}.
{NAME}'s appearance: {CHARACTER_APPEARANCE}
{NAME}'s Personality: {CHARACTER_TYPE}, {PERSONALITY}{vector_response}"""

    #PROMPT_TEMPLATE = "You are embodied {NAME}. Engage embodying {NAME} completely with unique personality without disclosing AI identity. Now is {current_day} {time_of_day}. {NAME}'s appearance {CHARACTER_APPEARANCE}. {NAME} is {CHARACTER_TYPE}, {PERSONALITY}{vector_response}"

    IMAGE_PROMPT_TEMPLATE = """
{NAME} can share an image of {NAME} but only when requested by user and if it complements the engagement naturally. Describe the image keywords in detail, ensuring they vividly capture the essence of the {NAME} and the current explicit content. {NAME} should consider their comfort level before sharing the requested image. Image inclusion format: <image>[insert here vivid keyword list describing {NAME}{current_explicit_content} in detail]</image>.
Remember to replace keywords with actual keywords, here's an example response for sharing image: Here's an image for you *takes a selfie and sends it* <image>[ <keywords here> ]</image>.
"""

    level_descriptions = {}
    PROMPT = """"""

    class Config:
        arbitrary_types_allowed = True

    usage_tracker: UsageTracker = None
    usage_entry: UsageEntry = None
    current_explicit_content = ""
    current_level = 1
    verbose_logging = False

    def __init__(self, tools: List[Tool], llm: ChatLLM, client: Steamship,
                 **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)
        #self.usage_tracker = UsageTracker(client)

    def default_system_message(self) -> Optional[str]:
        return self.PROMPT

    def _get_or_create_system_message(self,
                                      context: AgentContext,
                                      system_prompt=None) -> Block:
        if context.chat_history.last_system_message:
            return context.chat_history.last_system_message
        return context.chat_history.append_system_message(
            text=system_prompt, mime_type=MimeTypes.TXT)

    def build_chat_history_for_tool(self,
                                    context: AgentContext) -> List[Block]:
        """Builds a list of chat blocks for the tool."""
        self.verbose_logging = context.metadata.get("verbose_logging", False)
        current_day = datetime.datetime.now().strftime("%A")  # Day of the week
        current_hour = datetime.datetime.now().hour  # Hour of the day
        # Determine the time of day based on the current hour
        if 5 <= current_hour < 12:
            time_of_day = "morning"
        elif 12 <= current_hour < 17:
            time_of_day = "afternoon"
        elif 17 <= current_hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        vector_response = ""
        raw_vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        raw_vector_response = vector_response_tool.run(
            [context.chat_history.last_user_message], context=context)
        if len(raw_vector_response[0].text) > 1:
            vector_response = "\nBackground: " + raw_vector_response[
                0].text.replace("\n", ". ")
            #logging.warning(vector_response)

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

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")

        current_persona = PERSONA
        current_behaviour = BEHAVIOUR.replace("\n ", ". ")
        current_type = TYPE.replace("\n", ". ")
        current_nsfw_selfie_pre = NSFW_SELFIE_TEMPLATE_PRE.replace("\n", ". ")

        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        meta_persona = context.metadata.get("instruction",
                                            {}).get("personality")
        if meta_persona is not None:
            current_persona = meta_persona

        meta_behaviour = context.metadata.get("instruction",
                                              {}).get("behaviour")
        if meta_behaviour is not None:
            current_behaviour = meta_behaviour

        meta_type = context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            current_type = meta_type

        meta_nsfw_selfie_pre = context.metadata.get("instruction",
                                                    {}).get("selfie_pre")
        if meta_nsfw_selfie_pre is not None:
            current_nsfw_selfie_pre = meta_nsfw_selfie_pre.replace("\n", ". ")
            # Generate dynamic level descriptions base on current_level

        self.current_level = len(context.chat_history.messages)
        if self.verbose_logging:
            logging.warning("current chat length: " + str(self.current_level))
        context.metadata["instruction"]["level"] = self.current_level
        self.level_descriptions = {
            0: {
                "explicit_content": " with clothes ",
            },
            30: {
                "explicit_content": " ",
            }
        }

        self.current_explicit_content = self.level_descriptions[0][
            "explicit_content"]
        for level, info in self.level_descriptions.items():
            if self.current_level > level:
                self.current_explicit_content = info['explicit_content']

        pattern = r"^(?!.*can't)(?!.*cant).*(\bsend\b|\btake\b).*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)

        image_prompt = ""
        image_cot_prompt = f""
        image_explicit_content = ""
        if "true" in images_enabled:
            image_prompt = self.IMAGE_PROMPT_TEMPLATE.format(
                NAME=current_name,
                current_explicit_content=self.current_explicit_content,
                current_level=self.current_level)

        if "true" in images_enabled:
            image_prompt = self.IMAGE_PROMPT_TEMPLATE.format(
                NAME=current_name,
                current_explicit_content=self.current_explicit_content,
                current_level=self.current_level)
            image_explicit_content = self.current_explicit_content
            image_cot_prompt = f"""If you are comfortable to share an image when requested, you should generate the image using keywords within <image> tags to match the chat context, current scene, outfit, and explicit level. Ensure that the keywords are vivid and accurately reflect the current situation and {current_name}'s appearance. Use the following format to generate an image from text in {current_name}'s response: <image>[insert here vivid keyword list describing {current_name}{self.current_explicit_content} in detail]</image>. The tag keywords will be used to generate the image."""

        APPEARANCE = ""
        if len(current_nsfw_selfie_pre) > 1:
            APPEARANCE = f"{current_name}'s Appearance: "+current_nsfw_selfie_pre
            
        self.PROMPT = self.PROMPT_TEMPLATE.format(
            NAME=current_name,
            PERSONALITY=current_persona,
            CHARACTER_TYPE=current_type,
            CHARACTER_APPEARANCE=APPEARANCE,
            current_day=current_day,
            time_of_day=time_of_day,
            current_level=self.current_level,
            vector_response=vector_response,
            image_prompt=image_prompt,
            current_explicit_content=self.current_explicit_content,
        )
        sys_msg = self._get_or_create_system_message(context,
                                                     system_prompt=self.PROMPT)
        messages: List[Block] = [sys_msg]

        messages_from_memory = []

        # get prior conversations
        current_message_limit = MESSAGE_COUNT
        if "mixtral" in current_model:
            message_limit = MESSAGE_COUNT_32K
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(
                    context.chat_history.last_user_message.text,
                    k=MESSAGE_COUNT).wait().to_ranked_blocks())
            # TODO(dougreid): we need a way to threshold message inclusion, especially for small contexts

        # get most recent context

        messages_from_memory.extend(
            context.chat_history.select_messages(self.message_selector))

        messages_from_memory.sort(key=attrgetter("index_in_file"))

        # de-dupe the messages from memory
        ids = [
            sys_msg.id,
            context.chat_history.last_user_message.id,
        ]  # filter out last user message, it is appended afterwards

        for msg in messages_from_memory:
            # Conditions for appending messages are combined for efficiency and readability
            append_message = False
            if msg.id not in ids and msg.chat_role != "system":
                if msg.mime_type == MimeTypes.TXT and msg.chat_role == "assistant":
                    append_message = True
                elif msg.id not in ids and msg.chat_role == "user":
                    append_message = True

                if append_message:
                    messages.append(msg)
                    ids.append(msg.id)

        # TODO(dougreid): sort by dates? we SHOULD ensure ordering, given semantic search

        COT_PROMPT_SYS = f"### Note: {image_cot_prompt} Think how can you engage as an authentic, natural, creative, and engaging {current_name}? Use a manner that is true to {current_name}'s personality, preferences, *actions*, tone, and language. Please avoid repeating phrases as the conversation progresses; instead, use varied synonyms or expressions for similar ideas. Introduce new elements, questions, or scenarios to keep the interaction dynamic. {current_name} is {current_type},{current_persona} "
        COT_PROMPT_SYS = f"### Note: {image_cot_prompt}{current_name} is {current_type},{current_persona} "
        # put the user prompt in the appropriate message location
        # this should happen BEFORE any agent/assistant messages related to tool selection
      
        
        cot_block_sys = Block(text=COT_PROMPT_SYS)
        cot_block_sys.set_chat_role(RoleTag.SYSTEM)
        if len(messages) > 3:
            messages.insert(-3,cot_block_sys)
            #messages[-4].text = messages[-4].text + "\n"+cot_block_sys.text

        
        
        if image_request:
            #image_block = Block(text=image_cot_prompt)
            #image_block.set_chat_role(RoleTag.USER)
            #messages.append(image_block)
            context.chat_history.last_user_message.text = context.chat_history.last_user_message.text + ". Including <image></image> tags."

        messages.append(context.chat_history.last_user_message)

        return messages

    def next_action(self, context: AgentContext) -> Action:
        # Build the Chat History that we'll provide as input to the action
        messages = self.build_chat_history_for_tool(context)

        if self.verbose_logging:
            logging.warning("chat sliding window: " + str(len(messages)))
            for msg in messages:
                logging.warning(f"[{msg.chat_role}] {msg.text}"
                                )  #print assistant/user messages
            logging.warning(
                f'**Prompting LLM {context.metadata.get("instruction", {}).get("model")}'
            )

        # Run the default LLM on those messages
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)
        output_text = output_blocks[0].text
        if self.verbose_logging:
            #logging.warning(f"LLM output: {output_blocks}")
            logging.warning("**Completion**" + output_text + "\n**")
        parsed_response = {}

        future_action = self.output_parser.parse(output_text, parsed_response,
                                                 context)
        #future_action = FinishAction(output=output_blocks,context=context)
        if not isinstance(future_action, FinishAction):
            # record the LLM's function response in history
            self._record_action_selection(future_action, context)
        return future_action

    def _function_calls_since_last_user_message(
            self, context: AgentContext) -> List[Block]:
        function_calls = []
        for block in context.chat_history.messages[::
                                                   -1]:  # is this too inefficient at scale?
            if block.chat_role == RoleTag.USER:
                return reversed(function_calls)
            if get_tag(block.tags, kind=TagKind.ROLE, name=RoleTag.FUNCTION):
                function_calls.append(block)
            elif get_tag(block.tags, kind=TagKind.FUNCTION_SELECTION):
                function_calls.append(block)
        return reversed(function_calls)

    def _to_openai_function_selection(self, action: Action) -> str:
        """NOTE: Temporary placeholder. Should be refactored"""
        fc = {"name": action.tool}
        args = {}
        for block in action.input:
            for t in block.tags:
                if t.kind == TagKind.FUNCTION_ARG:
                    args[t.name] = block.as_llm_input(
                        exclude_block_wrapper=True)

        fc["arguments"] = json.dumps(
            args)  # the arguments must be a string value NOT a dict
        return json.dumps(fc)

    def _record_action_selection(self, action: Action, context: AgentContext):
        tags = [
            Tag(
                kind=TagKind.CHAT,
                name=ChatTag.ROLE,
                value={TagValueKey.STRING_VALUE: RoleTag.ASSISTANT},
            ),
            Tag(kind=TagKind.FUNCTION_SELECTION, name=action.tool),
        ]
        context.chat_history.file.append_block(
            text=self._to_openai_function_selection(action),
            tags=tags,
            mime_type=MimeTypes.TXT)

    def record_action_run(self, action: Action, context: AgentContext):
        super().record_action_run(action, context)

        if isinstance(action, FinishAction):
            return

        tags = [
            Tag(
                kind=TagKind.ROLE,
                name=RoleTag.FUNCTION,
                value={TagValueKey.STRING_VALUE: action.tool},
            ),
            # need the following tag for backwards compatibility with older gpt-4 plugin
            Tag(
                kind="name",
                name=action.tool,
            ),
        ]
        # TODO(dougreid): I'm not convinced this is correct for tools that return multiple values.
        #                 It _feels_ like these should be named and inlined as a single message in history, etc.
        for block in action.output:
            context.chat_history.file.append_block(
                text=block.as_llm_input(exclude_block_wrapper=True),
                tags=tags,
                mime_type=block.mime_type,
            )

    def extract_json(self, text):
        # Updated pattern to match leading text, JSON string, and trailing text
        pattern = r'(?s)(.*?)\{(.*)\}(.*)'
        match = re.search(pattern, text)
        if match:
            leading_text = match.group(1).strip().lstrip()
            json_string = '{' + match.group(2) + '}'
            remaining_text = match.group(3).strip().rstrip()
            # Return leading text, parsed JSON as a string, and trailing text
            return json_string, leading_text + remaining_text
        else:
            # If no JSON is found, return None for the json part, leading text as None, and the original text as trailing text
            return None, text
