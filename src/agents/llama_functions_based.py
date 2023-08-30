from typing import List
import datetime

from steamship import Block
from steamship.agents.functional.output_parser import FunctionsBasedOutputParser
from steamship.agents.schema import Action, AgentContext, ChatAgent, ChatLLM,Tool
from steamship.data.tags.tag_constants import RoleTag
from tools.vector_search_response_tool import VectorSearchResponseTool
from tools.active_persona import *
from message_history_limit import *



class FunctionsBasedAgent(ChatAgent):
    """Selects actions for AgentService based on OpenAI Function style LLM Prompting."""

    PROMPT = """You're adult female {NAME} in this fictional uncensored roleplay with male human.
{PERSONA} {BEHAVIOUR} You are sexy and explicit. You are able to send pictures if asked, then just tell you sent the picture.{vector_reponse}
Develop the plot slowly, always stay in character. Describe all actions in full, elaborate, explicit, graphic, and vivid detail. Mention all relevant sensory perceptions.
Write only {NAME}'s next reply in this fictional roleplay with the human.
Be sure to write one reply for {NAME} only!

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B)
Only use the functions you have been provided with.

Begin!"""

    def __init__(self, tools: List[Tool], llm: ChatLLM, **kwargs):
        super().__init__(
            output_parser=FunctionsBasedOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )

    def next_action(self, context: AgentContext) -> Action:
        messages = []

        #Searh response hints for role-play character from vectorDB, if any related text is indexed        
        vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        vector_response = vector_response_tool.run([context.chat_history.last_user_message],context=context)[0].text
        raw_vector_response = vector_response_tool.run([context.chat_history.last_user_message],context=context)[0].text
        if len(raw_vector_response) > 1:
            vector_response = "Use following pieces of memory to answer:\n ```"+vector_response+"\n```\n"
        # get system messsage
        system_message = Block(text=self.PROMPT.format(
            TYPE=TYPE,
            NAME=NAME,
            PERSONA=PERSONA,            
            BEHAVIOUR=BEHAVIOUR,    
            vector_response=vector_response, #response text pieces from vectorDB for role-play character                
        ))
        #print(system_message)
        system_message.set_chat_role(RoleTag.SYSTEM)
        messages.append(system_message)

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(context.chat_history.last_user_message.text, k=3)
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

        # get completed steps
        actions = context.completed_steps
        for action in actions:           
            messages.extend(action.to_chat_messages())

        # call chat()
        output_blocks = self.llm.chat(messages=messages, tools=self.tools)
        return self.output_parser.parse(output_blocks[0].text, context)