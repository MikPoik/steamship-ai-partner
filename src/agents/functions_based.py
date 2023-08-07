from typing import List
import datetime

from steamship import Block
from steamship.agents.functional.output_parser import FunctionsBasedOutputParser
from steamship.agents.schema import Action, AgentContext, ChatAgent, ChatLLM, Tool
from steamship.data.tags.tag_constants import RoleTag
from tools.mood_tool import MoodTool
from tools.active_persona import *
from message_history_limit import *


class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""

    PROMPT = """You are a helpful AI assistant.

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B)

Only use the functions you have been provided with."""

    def __init__(self, tools: List[Tool], llm: ChatLLM, **kwargs):
        super().__init__(
            output_parser=FunctionsBasedOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )

    def next_action(self, context: AgentContext,vector_response: str = "",words_left = 0) -> Action:

        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")

        #Get current mood, if keywords found, append to input (not saved in history)
        mood = MoodTool()
        special_mood = mood.run([context.chat_history.last_user_message],context)
        special_mood = special_mood[0].text    

        messages = []

        respond_with_words = "Answer with a word limit of 150!"
        if words_left < 50 and words_left > 0:
            respond_with_words = "You MUST Answer with a word limit of "+str(words_left)+"!"


        # get system messsage
        system_message = Block(text=self.PROMPT.format(
            TYPE=TYPE,
            NAME=NAME,
            PERSONA=PERSONA,            
            current_time=current_time,
            current_date=current_date,
            current_day=current_day,
            BEHAVIOUR=BEHAVIOUR,    
            vector_response=vector_response, #response text pieces from vectorDB for role-play character
            answer_word_cap=respond_with_words,
            special_mood=special_mood, #mood prompt                    
        ))
        #print("system prompt")
        #print(system_message.text)

        system_message.set_chat_role(RoleTag.SYSTEM)
        messages.append(system_message)

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(context.chat_history.last_user_message.text, k=int(RELEVANT_MESSAGES))
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
        
        #print("memory messages")
        #for msg in messages:
        #    if  msg.chat_role == RoleTag.USER:
        #          print(msg.chat_role +": "  + msg.text+"\n")
        #    if  msg.chat_role == RoleTag.ASSISTANT:
        #            print(msg.chat_role +": "  + msg.text+"\n")

        # get completed steps
        actions = context.completed_steps
        for action in actions:
            messages.extend(action.to_chat_messages())

        # call chat()
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)

        return self.output_parser.parse(output_blocks[0].text, context)
