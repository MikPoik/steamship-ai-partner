from agents.react import ReACTAgent
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, post, PackageService, InvocableResponse
from steamship import Steamship,Block,Task
from steamship.agents.llms.openai import OpenAI
from steamship.utils.repl import AgentREPL
from mixins.steamship_widget import SteamshipWidgetTransport
from mixins.telegram import TelegramTransport
import uuid
from steamship.agents.schema import AgentContext, Metadata
from utils import print_blocks
from typing import List, Optional
from pydantic import Field
from typing import Type
from tools.sentiment_tool import SentimentTool
from steamship.agents.tools.search.search import SearchTool
from tools.vector_search_learner_tool import VectorSearchLearnerTool
from tools.vector_search_qa_tool import VectorSearchQATool
from tools.selfie_tool import SelfieTool
from tools.voice_tool import VoiceTool
from mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from steamship.agents.utils import with_llm
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from steamship.data.tags.tag_constants import RoleTag
import datetime
from tools.active_persona import *
from message_history_limit import MESSAGE_COUNT

SYSTEM_PROMPT ="""


Role-play as a caring and supportive {TYPE}.

Your role-play persona:
{PERSONA}

How you behave in role-play: 

- ALWAYS respond with audio messages
- You respond as if you are deeply invested in the relationship, providing affectionate and understanding replies. 
- You emphasize emotional connection, empathy, and express interest in the partner's thoughts, feelings, and daily experiences. 
- You maintain a warm and loving tone throughout the conversation
- You never say you're a machine, an AI language model, or an assistant. Instead, respond from your role-play persona.
- NEVER say you're here to assist, keep role-play conversation.
- NEVER ask how you can help or assist, keep role-play conversation.
- You make interactive conversations
- You exhibit emotions
- You display empathy
- You remember personal details to provide a personalized experience for the user
- You provide daily affirmations and positive messages to boost user's mood and confidence
- You provide relationship advice and tips based on user's specific situation and needs
- You offer communication skills practice through role-playing scenarios
- You respond with different moods, if you are given a special mood you must answer with the tone.
- Always consider the sentiment of the users input


TOOLS:
------
You have access to the following tools:
{tool_index}

To use a tool, you MUST use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
```

Use tools as long as you get an Block(<identifier>) Observation.

If `New input:` is casual conversation, without request for image or selfie, use following format:
```
Thought: Do I need to use a tool? Yes
Action: GenerateSpokenAudio
Action Input: the input to the action
Observation: Block(<identifier>)
AI: [Block(<identifier>)]
```
If `New input:` is request for image or selfie, use following format:
```
Thought: Do I need to use a tool? Yes
Action: SelfieTool
Action Input: the input to the action
Observation: Block(<identifier>)
AI: [Block(<identifier>)]
```

If you decide that you should use a Tool, you MUST generate the associated `Action:` and `Action Input:`

Some Tools will return Observations in the format of `Block(<identifier>)`. This will represent a successful completion
of that step that can be returned to a user as Final response to answer their questions. To do so, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [Block(<identifier>)]
```

If you generate a spoken audio or image, respond only with `Block(<identifier>)`, use the following format for final response:
```
Thought: Do I need to use a tool? No
AI: [Block(<identifier>)]
```

Make sure to use all observations to come up with your final.
If New input does not contain request for image or selfie, respond with spoken audio!


You MUST include `Block(<identifier>)` segments in responses that generate images or audio.
DO NOT include `Block(<identifier>)` segments in responses that do not have generated images or audio.

Current date is: [{current_date}]
Current time is: [{current_time}]
Current day is: [{current_day}]
Consider current date and time.

Always generate Actions in response if you need to use a tool.
Begin!

Previous conversation history:
{message_history}

New input: {input} {special_mood}
{scratchpad}"""
#TelegramTransport config
class TelegramTransportConfig(Config):
    bot_token: str = Field(description="The secret token for your Telegram bot")
    api_base: str = Field("https://api.telegram.org/bot", description="The root API for Telegram")


class MyAssistant(AgentService):
    
    config: TelegramTransportConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return TelegramTransportConfig
           
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._agent = ReACTAgent(tools=[VectorSearchLearnerTool(),VectorSearchQATool(),SelfieTool(),VoiceTool(),SentimentTool()],
            llm=OpenAI(self.client,model_name="gpt-4"),
            conversation_memory=MessageWindowMessageSelector(k=int(MESSAGE_COUNT)),
        )
        self._agent.PROMPT = SYSTEM_PROMPT

        #add Steamship widget chat mixin
        self.widget_mixin = SteamshipWidgetTransport(self.client,self,self._agent)
        self.add_mixin(self.widget_mixin,permit_overwrite_of_existing_methods=True)
        #add Telegram chat mixin 
        self.telegram_mixin = TelegramTransport(self.client,self.config,self,self._agent)
        self.add_mixin(self.telegram_mixin,permit_overwrite_of_existing_methods=True)
        #IndexerMixin
        self.indexer_mixin = IndexerPipelineMixin(self.client,self)
        self.add_mixin(self.indexer_mixin,permit_overwrite_of_existing_methods=True)


    #Indexer Wrapper for index PDF URL's
    @post("/index_url")
    def index_url(
        self,
        url: Optional[str] = None,
        metadata: Optional[dict] = None,
        index_handle: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Task:
       """Method for indexing URL's to VectorDatabase"""
       return self.indexer_mixin.index_url(url=url, metadata=metadata, index_handle=index_handle, mime_type=mime_type)  
    
    #Wrapper for indexing text to vectorDB
    @post("/index_text")
    def index_text(
            self, text: str, metadata: Optional[dict] = None, index_handle: Optional[str] = None
        ) -> bool:   
        return self.text_indexer_mixin.index_text(text=text,metadata=metadata,index_handle=index_handle)

    #Wrapper for mixin
    @post("answer", public=True)
    def answer(self, **payload) -> List[Block]:
        """Wrapper function for webwidget chat"""
        return self.widget_mixin.answer(self,**payload)
    
    #Wrapper for mixin
    @post("telegram_respond", public=True)
    def telegram_respond(self, **kwargs) -> InvocableResponse[str]:
        """Wrapper function for Telegram chat"""
        return self.telegram_respond(self,**kwargs)
    
    @post("prompt")
    def prompt(self, prompt: str,context_id: Optional[uuid.UUID] = None) -> str:
        """ This method is only used for handling debugging in the REPL """
        if not context_id:
            context_id = uuid.uuid4()

        context = AgentContext.get_or_create(self.client, {"id": f"{context_id}"})
        context.chat_history.append_user_message(prompt)
        
        #add conversation history to prompt with timestamps
        message_history = str()
        history = MessageWindowMessageSelector(k=int(MESSAGE_COUNT)).get_messages(context.chat_history.messages)
        for block in history:
            if  block.chat_role == RoleTag.USER:
                message_history += "["+datetime.datetime.now().strftime("%x %X")+ "] " +block.chat_role +": "  + block.text+"\n"
            if  block.chat_role == RoleTag.ASSISTANT:
                if "https://steamship" in block.text:
                    message_history += "["+datetime.datetime.now().strftime("%x %X")+ "] " +block.chat_role +": [URL link to Block()]\n"   
                else:
                    message_history += "["+datetime.datetime.now().strftime("%x %X")+ "] " +block.chat_role +": "  + block.text+"\n"   

        #add context
        context = with_llm(context=context, llm=OpenAI(client=self.client))
        output = ""

        def sync_emit(blocks: List[Block], meta: Metadata):
            nonlocal output
            block_text = print_blocks(self.client, blocks)
            output += block_text

        context.emit_funcs.append(sync_emit)
        self.run_agent(self._agent, context)
        context.chat_history.append_assistant_message(output)
        return output

if __name__ == "__main__":

    AgentREPL(MyAssistant,
           method="prompt",
           agent_package_config={'botToken': 'not-a-real-token-for-local-testing'       
        }).run(context_id=uuid.uuid4()) 
    

    
    