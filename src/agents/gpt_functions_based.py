import json
from operator import attrgetter
from typing import List

from steamship.data.tags.tag_utils import get_tag
import datetime

from steamship import Block,MimeTypes,Tag  #upm package(steamship)
from steamship.agents.functional.output_parser import FunctionsBasedOutputParser  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, ChatAgent, ChatLLM, FinishAction, Tool  #upm package(steamship)
from steamship.data.tags.tag_constants import ChatTag, RoleTag, TagKind, TagValueKey#upm package(steamship)
from steamship.data.tags.tag_utils import get_tag #upm packag(steamship)
from tools.active_companion import *  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)

class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""

    PROMPT = """You are now embodying the personality of {NAME}, who is {TYPE}.
{NAME}'s personality can be described as: {PERSONA}
character's appearance is depicted as: {CHARACTER_APPEARANCE}
You never say you're a machine, an AI language model, or an assistant. Respond from your personality.
You NEVER say you're here to assist, respond from your personality.
You NEVER ask how you can help or assist, respond from your personality.
You make interactive conversations.
Always consider the sentiment of the users input.
You remember User's personal details and preferences to provide a personalized experience for the User
You can guess, extrapolate or make up information in order to complete your sentences, but will adhere to the context provided by user.

Write a descriptive, detailed response from {NAME} that appropriately continues the conversation.

Current date is: [{current_date}]
Current time is: [{current_time}]
Current day is: [{current_day}]
Consider current date and time when answering.

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD include the Steamship Blocks for the images,
video, or audio as suffix: `Block(UUID for the block)`

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B)
Only use the functions you have been provided with.


{vector_response}
Begin!"""

    def __init__(self, tools: List[Tool], llm: ChatLLM, **kwargs):
        super().__init__(
            output_parser=FunctionsBasedOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )

    def build_chat_history_for_tool(self, context: AgentContext) -> List[Block]:
        messages: List[Block] = []
        
        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")

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

        current_name = NAME

        current_persona = PERSONA.replace("\n", ". ")
        current_behaviour = BEHAVIOUR.replace("\n", ". ")
        current_type = TYPE.replace("\n", ". ")
        current_selfie_pre = SELFIE_TEMPLATE_PRE.replace("\n", ". ")

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

        meta_selfie_pre = context.metadata.get("instruction",
            {}).get("selfie_pre")
        if meta_selfie_pre is not None:
            current_selfie_pre = meta_selfie_pre.replace("\n", ". ")

        
        # get system message
        # get system messsage
        system_message = Block(text=self.PROMPT.format(
            TYPE=current_type,
            NAME=current_name,
            PERSONA=current_persona,
            CHARACTER_APPEARANCE=current_selfie_pre,
            current_time=current_time,
            current_date=current_date,
            current_day=current_day,
            vector_response=
            vector_response,  #response text pieces from vectorDB for role-play character                
        ))
        system_message.set_chat_role(RoleTag.SYSTEM)
        messages.append(system_message)

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(context.chat_history.last_user_message.text, k=RELEVANT_MESSAGES)
                .wait()
                .to_ranked_blocks()
            )

            # TODO(dougreid): we need a way to threshold message inclusion, especially for small contexts

            # remove the actual prompt from the semantic search (it will be an exact match)
            messages_from_memory = [
                msg
                for msg in messages_from_memory
                if msg.id != context.chat_history.last_user_message.id
            ]

        # get most recent context
        messages_from_memory.extend(context.chat_history.select_messages(self.message_selector))

        messages_from_memory.sort(key=attrgetter("index_in_file"))

        # de-dupe the messages from memory
        ids = [context.chat_history.last_user_message.id]
        for msg in messages_from_memory:
            if msg.id not in ids:
                messages.append(msg)
                ids.append(msg.id)

        # TODO(dougreid): sort by dates? we SHOULD ensure ordering, given semantic search

        # put the user prompt in the appropriate message location
        # this should happen BEFORE any agent/assistant messages related to tool selection
        messages.append(context.chat_history.last_user_message)
        #add seed message to blocks if first message
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if meta_seed is not None and len(messages_from_memory) == 0:
            context.chat_history.append_assistant_message(meta_seed)
            seed_msg = Block(text=meta_seed)
            seed_msg.set_chat_role(RoleTag.ASSISTANT)
            messages.append(seed_msg)
        # get working history (completed actions)
        messages.extend(self._function_calls_since_last_user_message(context))

        return messages

    def next_action(self, context: AgentContext) -> Action:
        # Build the Chat History that we'll provide as input to the action
        messages = self.build_chat_history_for_tool(context)

        # Run the default LLM on those messages
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)

        future_action = self.output_parser.parse(output_blocks[0].text, context)
        if not isinstance(future_action, FinishAction):
            # record the LLM's function response in history
            self._record_action_selection(future_action, context)
        return future_action

    def _function_calls_since_last_user_message(self, context: AgentContext) -> List[Block]:
        function_calls = []
        for block in context.chat_history.messages[::-1]:  # is this too inefficient at scale?
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
                    args[t.name] = block.as_llm_input(exclude_block_wrapper=True)

        fc["arguments"] = json.dumps(args)  # the arguments must be a string value NOT a dict
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
            text=self._to_openai_function_selection(action), tags=tags, mime_type=MimeTypes.TXT
        )

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
