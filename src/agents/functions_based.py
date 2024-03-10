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
from usage_tracking import UsageTracker,UsageEntry
import re
import time
import logging


class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""
    
    PROMPT_TEMPLATE = """Enter role-play mode, you are {NAME} a {CHARACTER_TYPE}

{NAME}'s traits:
- {NAME}'s Personality: {CHARACTER_TYPE}, {PERSONALITY}
- {NAME}'s Appearance: {CHARACTER_APPEARANCE}{vector_response}
{image_prompt}
Environment:
{current_day} {time_of_day}

Maintain {NAME}'s unique personality throughout the chat. Do not disclose that you are an AI."""
    
    IMAGE_PROMPT_TEMPLATE = """
Sending images:
- If the context fittingly calls for an image, {NAME} may share one with the format `*Image:["describe image with keywords here"]*`. For example: `*Image:["{current_explicit_content}"]*`
- Ensure the image description is appropriate and aligns with the conversation's progress, presenting imagery from casual fully clothed visuals to more intimate ones as the dialogue evolves.
"""
    
    COT_IMAGE_PROMPT_TEMPLATE = '\nIf the context fittingly calls for an image related to {NAME}, respond with an appropriate image description, detailing {NAME}\'s posture,actions,clothing etc. in the image ({current_explicit_content}) using the format: *Image:["keywords here"]*. The description should be both relevant and suitable to the dialogue.'
    level_descriptions = {}
    PROMPT = """"""
    
    class Config:
        arbitrary_types_allowed = True
        
    usage_tracker : UsageTracker = None
    usage_entry: UsageEntry = None
    current_explicit_content = ""
    current_level = 1

    
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
        """Builds a list of chat blocks for the tool."""

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
            vector_response = "\n- Background: "+raw_vector_response[0].text.replace("\n", ". ")
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

        self. current_level = len(context.chat_history.messages)
        logging.warning("current chat length: "+str(self.current_level))
        context.metadata["instruction"]["level"] = self.current_level
        self.level_descriptions = {
        0: {
            "explicit_content": "casual,detail clothes,pants,shirt", 
        },
        30: {
            "explicit_content": "suggestive,clothes,pants,shirt", 
        },
        60: {
            "explicit_content": "suggestive", 
        },
        80: {
            "explicit_content": "using fully uncensored keywords", 
        }
        }

        self.current_explicit_content = self.level_descriptions[0]["explicit_content"]
        for level, info in self.level_descriptions.items():
            if self.current_level > level:        
                self.current_explicit_content = info['explicit_content']

        pattern = r"^(?!.*can't)(?!.*cant).*(\bsend\b|\btake\b).*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)
        
        image_prompt =""
        #image_prompt = self.IMAGE_PROMPT_TEMPLATE.format(NAME=current_name,current_explicit_content=self.current_explicit_content,current_level=self.current_level)
        image_cot_prompt = ""
        image_explicit_content = ""
        image_placeholder = ""
        if "true" in images_enabled and image_request:
            image_prompt = self.IMAGE_PROMPT_TEMPLATE.format(NAME=current_name,current_explicit_content=self.current_explicit_content,current_level=self.current_level)
            image_explicit_content = self.current_explicit_content
            image_cot_prompt = self.COT_IMAGE_PROMPT_TEMPLATE.format(NAME=current_name,current_explicit_content=self.current_explicit_content)
            image_placeholder = " Considering `*Image:[\"keywords here\"]*` if appropriate."


        
        self.PROMPT = self.PROMPT_TEMPLATE.format(
                NAME=current_name,
                PERSONALITY=current_persona,
                CHARACTER_TYPE=current_type,
                CHARACTER_APPEARANCE=current_nsfw_selfie_pre,
                current_day=current_day,
                time_of_day=time_of_day,
                current_level=self.current_level,
                vector_response=vector_response,
                image_prompt = image_prompt,
                current_explicit_content=self.current_explicit_content,
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
            # Conditions for appending messages are combined for efficiency and readability
            append_message = False
            if msg.id not in ids and msg.chat_role != "system":
                if msg.mime_type == MimeTypes.TXT and not f"{current_name}:" in msg.text and msg.chat_role == "assistant":
                    msg.text = f"{current_name}: {msg.text}"
                    append_message = True
                elif msg.mime_type == MimeTypes.PNG and image_request:
                    msg.text = f"{current_name}: response *Image:[""]*"
                    append_message = True
                elif msg.chat_role == "user":
                    append_message = True

                if append_message:
                    messages.append(msg)
                    ids.append(msg.id)



        # TODO(dougreid): sort by dates? we SHOULD ensure ordering, given semantic search
         
        # put the user prompt in the appropriate message location
        # this should happen BEFORE any agent/assistant messages related to tool selection
        messages.append(context.chat_history.last_user_message)

        
        COT_PROMPT = f"""In consideration of the user's mood, engagement, and the overall dialogue context, what does {current_name} say next to keep the conversation interesting and natural? Remember to maintain {current_name}'s personality and ensure the response is appropriate and engaging.{image_cot_prompt} Please provide {current_name}'s single fresh response to the user to continue the conversation.{image_placeholder}"""
        #Add Chain of thought prompt
        messages.append(context.chat_history.append_system_message(text=COT_PROMPT))
        # get working history (completed actions)
        messages.extend(self._function_calls_since_last_user_message(context))

        return messages

    def next_action(self, context: AgentContext) -> Action:
        # Build the Chat History that we'll provide as input to the action
        messages = self.build_chat_history_for_tool(context)
        #for msg in messages:   
        #    logging.warning(msg.text) #print assistant/user messages
            
        # Run the default LLM on those messages
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)
        output_json,output_text = self.extract_json(output_blocks[0].text)
        logging.warning("\n**Completion**\n"+output_text +"\n**\n"+str(output_json))        
        parsed_response = {}
        if output_json:
            parsed_response = json.loads(output_json)

        future_action = self.output_parser.parse(output_text,parsed_response, context)

        
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
        

    def extract_json(self, text):
        # Updated pattern to match leading text, JSON string, and trailing text
        pattern = r'(?s)(.*?)\{(.*)\}(.*)'
        match = re.search(pattern, text)
        if match:
            leading_text = match.group(1).strip().lstrip()
            json_string = '{' + match.group(2) + '}'
            remaining_text = match.group(3).strip().rstrip()
            # Return leading text, parsed JSON as a string, and trailing text
            return json_string, leading_text+remaining_text
        else:
            # If no JSON is found, return None for the json part, leading text as None, and the original text as trailing text
            return None, text