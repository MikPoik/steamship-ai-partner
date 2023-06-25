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
from tools.vector_search_learner_tool import VectorSearchLearnerTool
from tools.vector_search_qa_tool import VectorSearchQATool
from tools.selfie_tool import SelfieTool
from tools.voice_tool import VoiceTool
from mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from steamship.agents.utils import with_llm
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.did_video_generator_tool import DIDVideoGeneratorTool
from tools.active_persona import *
from message_history_limit import MESSAGE_COUNT
from default_prompt_template import DEFAULT_TEXT_PROMPT

#PROMPTS are in *_prompt_template.py files.
#you can change default response style by saying 'default text' or 'default voice'
SYSTEM_PROMPT = DEFAULT_TEXT_PROMPT

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

        self._agent = ReACTAgent(tools=[VectorSearchLearnerTool(),VectorSearchQATool(),SelfieTool(),VoiceTool(),DIDVideoGeneratorTool()],
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
    

    
    