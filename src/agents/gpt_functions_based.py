from typing import List
import datetime

from steamship import Block #upm package(steamship)
from steamship.agents.functional.output_parser import FunctionsBasedOutputParser #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, ChatAgent, ChatLLM,Tool #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag #upm package(steamship)
from tools.mood_tool import MoodTool #upm package(steamship)
from tools.active_companion import * #upm package(steamship)
from message_history_limit import * #upm package(steamship)
from tools.vector_search_response_tool import VectorSearchResponseTool #upm package(steamship)

class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""

    PROMPT = """You are now embodying the personality of {NAME}, {TYPE}.
{PERSONA}
{BEHAVIOUR}
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
Begin!
"""

    def __init__(self, tools: List[Tool], llm: ChatLLM, **kwargs):
        super().__init__(
            output_parser=FunctionsBasedOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )

    def next_action(self, context: AgentContext) -> Action:
        messages = []
        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")

        #Searh response hints for role-play character from vectorDB, if any related text is indexed        
        vector_response = ""
        raw_vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        raw_vector_response = vector_response_tool.run([context.chat_history.last_user_message], context=context)
        if len(raw_vector_response) > 1:
            vector_response = raw_vector_response
            vector_response = "Use following pieces of memory to answer:\n ```" + vector_response + "\n```\n"


        current_name = NAME       
        current_persona = PERSONA
        current_behaviour = BEHAVIOUR
        current_type = TYPE  

        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name 

        meta_persona = context.metadata.get("instruction", {}).get("personality")
        if meta_persona is not None:
            current_persona = meta_persona

        meta_behaviour =  context.metadata.get("instruction", {}).get("behaviour")
        if meta_behaviour is not None:
            current_behaviour = meta_behaviour

        meta_type =  context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            current_type = meta_type
            
        # get system messsage
        system_message = Block(text=self.PROMPT.format(
            TYPE=current_type,
            NAME=current_name,
            PERSONA=current_persona,            
            BEHAVIOUR=current_behaviour,
            current_time=current_time,
            current_date=current_date,
            current_day=current_day,    
            vector_response=vector_response, #response text pieces from vectorDB for role-play character                
        ))
        #print(system_message)
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

        #add seed message to blocks if first message
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        if meta_seed is not None and len(messages_from_memory) == 0:
            context.chat_history.append_assistant_message(meta_seed)
            seed_msg = Block(text=meta_seed)
            seed_msg.set_chat_role(RoleTag.ASSISTANT)
            messages.append(seed_msg)

        # de-dupe the messages from memory
        ids = []
        for msg in messages_from_memory:
            if msg.id not in ids:
                messages.append(msg)
                ids.append(msg.id)

        # TODO(dougreid): sort by dates? we SHOULD ensure ordering, given semantic search

        # put the user prompt in the appropriate message location
        # this should happen BEFORE any agent/assistant messages related to tool selection
        messages.append(context.chat_history.last_user_message)

        # get completed steps
        actions = context.completed_steps
        for action in actions:
            #print(action.to_chat_messages())
            messages.extend(action.to_chat_messages())
        #print(messages)
        # call chat()
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)
        return self.output_parser.parse(output_blocks[0].text, context)
