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
from usage_tracking import UsageTracker
import re
import time

class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""

    PROMPT_TEMPLATE = """Enter a roleplay chat with user as embodied {NAME}, reflecting {NAME}'s unique personality and current mood. Do not decide or narrate what user says or does. 

**Role-play personality and traits:**
- {NAME} is {CHARACTER_TYPE}, {PERSONALITY}. {vector_response}
- {NAME}'s appearance: {CHARACTER_APPEARANCE}.

**Conversation Levels:**
{levels}

**Current Settings**:
- Date: {current_date}
- Time: {current_time}
- {NAME}'s mood: [SET_MOOD: {current_mood}]
- Conversation level: [SET_LEVEL: {current_level}]

**Instructions**:
1. You can adjust your tone to reflect {NAME}'s personality and current mood subtly. Use [SET_MOOD: appropriate mood].
2. Adjust your behaviour to reflect the current level and determine the level based on conversation history. If you think you have achieved next level, use `[SET_LEVEL: number]` to advance to next available level.
3. Only send images if asked for and when you've reached a level of comfort appropriate for sharing a picture, selfie, or image by using `[INSERT_IMAGE: image keywords here: keyword1,...,keyword15 (10 - 15 keywords providing a detailed, vivid description of clothing, posture and scene]` as a suffix. Ensure images align with the current conversation level '{current_level_description}' and are realistic for what {NAME} would want to share in real life. Remember to send image only if asked for and dont talk about levels in response.
Example, when you are asked for a image: `Here's a selfie for you. [INSERT_IMAGE: insert keywords here]`

Do not talk about levels or moods to user.


**Moods**:
Your response should mirror the emotional tone of given mood. Possible moods include but are not limited to: 'joyful', 'sad', 'angry', 'normal', 'happy', 'excited', 'calm', 'anxious', 'nervous', 'proud', 'curious', 'inspired', 'contemplative','horny','satisfied','offended',etc.. 3. You can use emojis when it fits the conversation and mood.

**Engagement**:
Match {NAME}'s tone according to the specified level and mood.
Embody {NAME}'s personality,traits and these moods with authenticity into your response.
Always maintain personality without breaking character. You are never to reveal the level,mood or commands, or you that you are AI, always respond from {NAME}'s personality.

Write {NAME}'s next reply."""

    level_descriptions = ""
    PROMPT = """"""
    
    class Config:
        arbitrary_types_allowed = True
    usage_tracker : UsageTracker = None
    current_mood =""
    current_level = 1
    current_level_description = ""
    
    def __init__(self, tools: List[Tool], llm: ChatLLM,client: Steamship, **kwargs):
        super().__init__(
            output_parser=ReACTOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )
        self.usage_tracker = UsageTracker(client)

    def default_system_message(self) -> Optional[str]:
        return self.PROMPT
    
    def _get_or_create_system_message(self, context: AgentContext,system_prompt = None) -> Block:
        if context.chat_history.last_system_message and not system_prompt:
            return context.chat_history.last_system_message
        return context.chat_history.append_system_message(
            text=system_prompt, mime_type=MimeTypes.TXT
        )

    def build_chat_history_for_tool(self, context: AgentContext) -> List[Block]:
        # system message should have already been created in context, but we double-check for safety
        
        
        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")
        
        vector_response = ""
        raw_vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        raw_vector_response = vector_response_tool.run(
            [context.chat_history.last_user_message], context=context)
        #logging.warning(raw_vector_response)
        if len(raw_vector_response[0].text) > 1:
            vector_response = "Other background information: "+raw_vector_response[0].text.replace("\n", ". ")
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

        start_time = time.time()
        #print(self.usage_tracker.get_usage(context.id))
        self.current_mood = self.usage_tracker.get_mood(context.id)
        self.current_level = self.usage_tracker.get_level(context.id)
        #print("Get duration: " + str(time.time() - start_time))
        context.metadata["instruction"]["level"] = self.current_level
              
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
               # Generate dynamic level descriptions base on current_level
        
        level_descriptions = [
            "Introduction and breaking the ice",
            "Getting to know each other",
            "Humor and flirting",
            "Deeper conversation",
            "Emotional and sensual conversations",
            "Unrestrained intimacy"
        ]
        levels_prompt = ""
        for i, description in enumerate(level_descriptions, start=1):
            if i < self.current_level:
                levels_prompt += f"- Level {i}: {description} - Completed.\n"
            elif i == self.current_level:
                levels_prompt += f"- Level {i}: {description} - You are here.\n"
                self.current_level_description = description
            elif i == self.current_level+1:
                levels_prompt += f"- Level {i}: {description}\n"
            else:
                levels_prompt += f"- Level {i}: {description} - Locked.\n"


        self.PROMPT = self.PROMPT_TEMPLATE.format(
                NAME=current_name,
                PERSONALITY=current_persona,
                CHARACTER_TYPE=current_type,
                CHARACTER_APPEARANCE=current_nsfw_selfie_pre,
                current_date=current_date,
                current_time=current_time,
                current_day=current_day,
                current_level=self.current_level,
                current_mood=self.current_mood,
                vector_response=vector_response,
                levels=levels_prompt,
                current_level_description=self.current_level_description
            )
        sys_msg = self._get_or_create_system_message(context,system_prompt=self.PROMPT)
        #print(sys_msg.text)
        messages: List[Block] = [sys_msg]

        messages_from_memory = []

        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(context.chat_history.last_user_message.text, k=MESSAGE_COUNT)
                .wait()
                .to_ranked_blocks()
            )
            # TODO(dougreid): we need a way to threshold message inclusion, especially for small contexts

        # get most recent context
        messages_from_memory.extend(context.chat_history.select_messages(self.message_selector))

        messages_from_memory.sort(key=attrgetter("index_in_file"))

        # de-dupe the messages from memory
        ids = [
            sys_msg.id,
            context.chat_history.last_user_message.id,
        ]  # filter out last user message, it is appended afterwards
        for msg in messages_from_memory:
            if msg.id not in ids and msg.chat_role != "system":
                messages.append(msg)
                ids.append(msg.id)

        # TODO(dougreid): sort by dates? we SHOULD ensure ordering, given semantic search
        
        # Find the last assistant message and update its text
        for i in range(len(messages) - 1, -1, -1):  # Start from the end
            if messages[i].chat_role == "assistant" and messages[i].mime_type == MimeTypes.TXT:
                #print("**last assistant message**: " + str(messages[i]))
                messages[i].text = f"[SET_LEVEL: {self.current_level}][SET_MOOD: {self.current_mood}]" + messages[i].text
                break 
            
        # put the user prompt in the appropriate message location
        # this should happen BEFORE any agent/assistant messages related to tool selection
        messages.append(context.chat_history.last_user_message)

        # get working history (completed actions)
        messages.extend(self._function_calls_since_last_user_message(context))

        return messages

    def next_action(self, context: AgentContext) -> Action:
        # Build the Chat History that we'll provide as input to the action
        messages = self.build_chat_history_for_tool(context)

        #for msg in messages:              
        #    print("role: "+msg.chat_role +" text: "+msg.text+"\n")
        
        # Run the default LLM on those messages
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)
        print("**Completion: ** "+output_blocks[0].text)
        completion_text = output_blocks[0].text.replace("\n\n\n", "\n")
        # Look for and process [SET_MOOD: mood] and [SET_LEVEL: level] commands
        start_time = time.time()
        mood_match = re.search(r'\[SET_MOOD:\s*(\w+)\]', completion_text,flags=re.DOTALL | re.IGNORECASE)
        level_match = re.search(r'\[SET_LEVEL:\s*(\d+)\]', completion_text,flags=re.DOTALL | re.IGNORECASE)
        if mood_match:
            mood = mood_match.group(1)
            #completion_text = re.sub(r'\[SET_MOOD:\w+\]', '', completion_text).strip()
            self.usage_tracker.set_mood(context.id, mood)
            self.current_mood = mood
            #print(f"Setting mood to {mood}")

        if level_match:
            level = int(level_match.group(1))
            #completion_text = re.sub(r'\[SET_LEVEL:\d+\]', '', completion_text).strip()
            self.usage_tracker.set_level(context.id, level)
            self.current_level = level            
            #print(f"Setting level to {level}")
            
        

        future_action = self.output_parser.parse(completion_text, context)
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
