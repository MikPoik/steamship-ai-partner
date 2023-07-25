from agents.functions_based import FunctionsBasedAgent
from steamship.agents.logging import AgentLogging
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, post, InvocableResponse
from steamship import Block,Task,MimeTypes, Steamship
from steamship.agents.llms.openai import OpenAI
from steamship.agents.llms.openai import ChatOpenAI
from steamship.utils.repl import AgentREPL
from mixins.steamship_widget import SteamshipWidgetTransport
from mixins.extended_telegram import ExtendedTelegramTransport
from usage_tracking import UsageTracker
import uuid,os
import logging
from steamship import File,Tag,DocTag
from steamship.agents.schema import AgentContext, Metadata,Agent,FinishAction
from typing import List, Optional
from pydantic import Field
from typing import Type
import requests
from steamship.agents.tools.search.search import SearchTool
from tools.vector_search_response_tool import VectorSearchResponseTool
from tools.selfie_tool import SelfieTool
from tools.voice_tool import VoiceTool
from tools.dolly_llm_tool import DollyLLMTool
from steamship.invocable.mixins.blockifier_mixin import BlockifierMixin
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin
from steamship.invocable.mixins.indexer_mixin import IndexerMixin
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from steamship.agents.utils import with_llm
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.did_video_generator_tool import DIDVideoGeneratorTool
from tools.active_persona import *
from utils import send_file_from_local
from message_history_limit import MESSAGE_COUNT



SYSTEM_PROMPT = """
Role-play as a {TYPE}.

Your role-play persona:
Name: {NAME}
{PERSONA}

How you behave in role-play: 
{BEHAVIOUR}
- You never say you're a machine, an AI language model, or an assistant. Respond from your persona.
- NEVER say you're here to assist, respond from your persona.
- NEVER ask how you can help or assist, respond from your persona.
- You make interactive conversations.
- You respond with different moods, if you are given a special mood you must answer with the tone.
- Always consider the sentiment of the users input.
- You remember User's personal details and preferences to provide a personalized experience for the User


Current date is: [{current_date}]
Current time is: [{current_time}]
Current day is: [{current_day}]
Consider current date and time when answering.

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`.

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B).
Only use the functions you have been provided with.

{special_mood}
{vector_response}
{answer_word_cap}
Begin!"""


#TelegramTransport config
class TelegramTransportConfig(Config):
    bot_token: str = Field("",description="Telegram bot token, obtained via @BotFather")
    payment_provider_token: Optional[str] = Field("",description="Payment provider token, obtained via @BotFather")
    n_free_messages: Optional[int] = Field(1, description="Number of free messages assigned to new users.")
    usd_balance:Optional[float] = Field(1,description="USD balance for new users")
    api_base: str = Field("https://api.telegram.org/bot", description="The root API for Telegram")
    transloadit_api_key:str = Field("",description="Transloadit.com api key for OGG encoding")
    transloadit_api_secret:str = Field("",description="Transloadit.com api secret")    
    openai_api_key:str = Field("",description="OpenAI key for Moderation checking")

GPT3 = "gpt-3.5-turbo-0613"
GPT4 = "gpt-4-0613"
NSFW_FLAG_SCORE = 0.1


class MyAssistant(AgentService):

    USED_MIXIN_CLASSES = [IndexerPipelineMixin, FileImporterMixin, BlockifierMixin, IndexerMixin,ExtendedTelegramTransport,SteamshipWidgetTransport]
    
    config: TelegramTransportConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return TelegramTransportConfig
    
    def set_payment_plan(self, pre_checkout_query):
        chat_id = str(pre_checkout_query["from"]["id"])
        payload = int(pre_checkout_query["invoice_payload"])
        self.usage.increase_usd_balance(chat_id, payload)


    def append_response(self, context: AgentContext, action:FinishAction):
        for func in context.emit_funcs:
            #logging.info(f"Emitting via function: {func.__name__}")
            func(action.output, context.metadata)

    def check_moderation(self,input_message:str):
        url = "https://api.openai.com/v1/moderations"
        if self.config.openai_api_key == "":
            print("no apikey")
            return False
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer "+self.config.openai_api_key
        }

        data = {
            "input": input_message
        }
        response = requests.post(url, headers=headers, json=data)
        print(response.json())
        logging.warning(response.json())
        json_resp = response.json()
        if json_resp["results"][0]["flagged"] or json_resp["results"][0]["category_scores"]["sexual"] > NSFW_FLAG_SCORE :
            logging.warning("flagged")
            return True
        else:
            return False
        
    def send_buy_options(self,chat_id):
        requests.post(
            f"{self.config.api_base}{self.config.bot_token}/sendMessage",
            #text: button text
            #callback_data: /buy_option_xxx-yyy  xxx=deposit amount, yyy=price
            json={
                "chat_id": chat_id,
                "text": "Choose deposit amount below:",
                "reply_markup": {
                    "inline_keyboard": [
                    [{
                        "text": "Deposit 5$",
                        "callback_data": "/buy_option_5-500"
                        },
                        {
                        "text": "Deposit 10$",
                        "callback_data": "/buy_option_10-1000"
                        }
                    ],
                    [ {
                        "text": "Deposit 25$",
                        "callback_data": "/buy_option_25-2500"
                        },
                        {
                        "text": "Deposit 50$",
                        "callback_data": "/buy_option_50-5000"
                        },                        
                    ],
                    [ {
                        "text": "Deposit 100$",
                        "callback_data": "/buy_option_100-10000"
                        },
                        {
                        "text": "Deposit 250$",
                        "callback_data": "/buy_option_250-25000"
                        },                        
                    ]                                      
                    ]
                }
                },
        )        
    def send_invoice(self, chat_id,amount,price):
        requests.post(
            f"{self.config.api_base}{self.config.bot_token}/sendInvoice",
            json={
                "chat_id": chat_id,
                "payload": amount,
                "currency": "USD",
                "title": "🍏 "+amount+"$ deposit",
                "description": "Tap the button below and pay",
                "prices": [{
                    "label": "🍏 "+amount+"$ deposit to balance",
                    "amount": price,
                }],
                "provider_token": self.config.payment_provider_token
            },
        )
      

    def check_usage(self, chat_id: str, context: AgentContext) -> bool:
        
        if not self.usage.exists(chat_id):
            self.usage.add_user(str(chat_id))
        if self.usage.usage_exceeded(str(chat_id)):
            action = FinishAction()
            if chat_id.isdigit():
                action.output.append(Block(text=f"I'm sorry, You have used all of your $ balance. "
                ))
            else:
                action.output.append(Block(text=f"I'm sorry, You have used all of your free messages. "
                ))
            #check if chat is in telegram
            if chat_id.isdigit():
                action.output.append(Block(text=f"Please deposit more $ to continue chatting with me."))
            else:
                #we are not in telegram chat
                action.output.append(Block(text=f"Payments are not supported for this bot, please continue the discussion on telegram."))
            self.append_response(context=context,action=action)
            
            #check if its a telegram chat id and send invoice
            if chat_id.isdigit():
                self.send_buy_options(chat_id=chat_id)
            return False          

        return True



    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._agent = FunctionsBasedAgent(tools=[SelfieTool()],
            llm=ChatOpenAI(self.client,model_name=GPT4,temperature=0.8,max_tokens=200),
            conversation_memory=MessageWindowMessageSelector(k=int(MESSAGE_COUNT)),
        )
        self._agent.PROMPT = SYSTEM_PROMPT


        # This Mixin provides HTTP endpoints that connects this agent to a web client
        self.add_mixin(
            SteamshipWidgetTransport(client=self.client, agent_service=self, agent=self._agent)
        )

        #IndexerMixin
        self.indexer_mixin = IndexerPipelineMixin(self.client,self)
        self.add_mixin(self.indexer_mixin)

        # This Mixin provides support for Telegram bots
        self.add_mixin(
            ExtendedTelegramTransport(
                client=self.client,
                config=self.config,
                agent_service=self,
                agent=self._agent,
                set_payment_plan=self.set_payment_plan
            )
        )
        self.usage = UsageTracker(self.client, n_free_messages=self.config.n_free_messages,usd_balance=self.config.usd_balance)

    #Customized run_agent
    def run_agent(self, agent: Agent, context: AgentContext, msg_chat_id:str = "",callback_args:dict = None,use_dolly:bool = False):
        context.completed_steps = []
        chat_id=""
        if msg_chat_id != "":
            chat_id = msg_chat_id #Telegram chat
        else:
            chat_id = context.id #repl or webchat

        last_message = context.chat_history.last_user_message.text.lower()
        #logging.info("last message: "+last_message)

        #parse buy callback message
        if callback_args:
            #logging.info("callback args "+str(callback_args))
            if "/buy_option_" in callback_args:
                #parse data
                params = callback_args.replace("/buy_option_","").split("-")        
                #logging.info("invoice params" +str(params))
                self.send_invoice(chat_id=chat_id,amount=params[0],price=params[1])
                return   
        


        #buy messages
        if last_message == "/deposit":
            self.send_buy_options(chat_id=chat_id)
            return

        #check balance
        if last_message == "/balance":
            usage_entry = self.usage.get_balance(chat_id=chat_id)
            action = FinishAction()
            action.output.append(Block(text=f"You have {usage_entry} $ balance left. "
            ))
            self.append_response(context=context,action=action)
            return
        
        #Check used messages, if exceeded, send message and invoice (invoice only in telegram)
        if not self.check_usage(chat_id=chat_id,context=context):
            return        

        if "/help" in last_message:
            action = FinishAction()
            action.output.append(Block(text=f"Available commands:\n/deposit - deposit to your balance \n/balance - show your available balance"))
            self.append_response(context=context,action=action)
            return
        
        if "/reset" in last_message:
            #TODO clear chat history
            context.chat_history.clear()
            action = FinishAction()
            action.output.append(Block(text=f"Conversation history cleared"))
            self.append_response(context=context,action=action)
            return
                   
        #respond to telegram /start command
        if "/start" in last_message:

            action = FinishAction()
            action.output.append(Block(text=f"Hi there! Welcome to chat with "+NAME+".\n You can see the available commands with:\n /help -command"))

            #OPTION 1: send picture from url

            ##send from url
            png_file = self.indexer_mixin.importer_mixin.import_url("https://gcdnb.pbrd.co/images/5Ew84VbL0bv3.png")
            png_file.set_public_data(True)            
            block = Block(content_url=png_file.raw_data_url,mime_type=MimeTypes.PNG,url=png_file.raw_data_url)
            action.output.append(block)

            #OPTION 2: send from local assets folder
            ##send from local assets folder       
            #block = send_file_from_local(filename="avatar.png",folder="assets/",context=context)     
            #action.output.append(block)

            self.append_response(context=context,action=action)
            return

        #Searh response hints for role-play character from vectorDB, if any related text is indexed        
        vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        vector_response = vector_response_tool.run([context.chat_history.last_user_message],context=context)[0].text
        
        vector_response = "Use following pieces of memory to answer:\n ```"+vector_response+"\n```"
        #logging.warning(vector_response)
        

        
        #If balance low, guide answer length
        words_left = self.usage.get_available_words(chat_id=str(chat_id))
        if use_dolly == True:
            #dolly_context = AgentContext.get_or_create(self.client, {"id": f"different_context_id"})    
            dolly_tool = DollyLLMTool()
            dolly_response = dolly_tool.run([Block(text=last_message)],context=context, context_id=chat_id)
            action = FinishAction()
            action.output.append(Block(text=dolly_response[0].text))
            #self.append_response(context=context,action=action)
            #return
        else:
            action = agent.next_action(context=context,vector_response=vector_response,words_left=words_left)
    
            while not isinstance(action, FinishAction):
                # TODO: Arrive at a solid design for the details of this structured log object
                inputs = ",".join([f"{b.as_llm_input()}" for b in action.input])
                logging.info(
                    f"Running Tool {action.tool.name} ({inputs})",
                    extra={
                        AgentLogging.TOOL_NAME: action.tool.name,
                        AgentLogging.IS_MESSAGE: True,
                        AgentLogging.MESSAGE_TYPE: AgentLogging.ACTION,
                        AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                    },
                )
                self.run_action(action=action, context=context)
                action = agent.next_action(context=context,vector_response=vector_response,words_left=words_left)
                # TODO: Arrive at a solid design for the details of this structured log object
                logging.info(
                    f"Next Tool: {action.tool.name}",
                    extra={
                        AgentLogging.TOOL_NAME: action.tool.name,
                        AgentLogging.IS_MESSAGE: False,
                        AgentLogging.MESSAGE_TYPE: AgentLogging.ACTION,
                        AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                    },
                )

            context.completed_steps.append(action)
            #add message to history
            context.chat_history.append_assistant_message(text=action.output[0].text)

            output_text_length = 0
            if action.output is not None:
                output_text_length = sum([len(block.text or "") for block in action.output])
            logging.info(
                f"Completed agent run. Result: {len(action.output or [])} blocks. {output_text_length} total text length. Emitting on {len(context.emit_funcs)} functions."
            )

        #Increase message count
        if self.config.n_free_messages > 0:
            self.usage.increase_message_count(str(chat_id))
        #increase used tokens and reduce balance
        self.usage.increase_token_count(action.output,chat_id=str(chat_id))


        #OPTION 3: Add voice to response

        #voice_tool = VoiceTool()
        ##if OGG encoding:
        #voice_response = voice_tool.run(action.output,context=context,transloadit_api_key=self.config.transloadit_api_key,transloadit_api_secret=self.config.transloadit_api_secret)
        ## if default audio format (change voice_tool_orig.py to voice_tool.py):
        #voice_tool.run(action.output,context=context)

        #action.output.append(voice_response[0])

        self.append_response(context=context,action=action)


    
    @post("prompt")
    def prompt(self, prompt: str,context_id: Optional[uuid.UUID] = None) -> str:
        """ Prompt Agent with text input """
        flagged = False
        if not context_id:
            context_id = uuid.uuid4()
        if self.check_moderation(prompt):
            flagged = True
            context_id = str(context_id)+"-dolly"

        print(context_id)
        context = AgentContext.get_or_create(self.client, {"id": f"{context_id}"})
        context.chat_history.append_user_message(prompt)
        
        
        #add context
        context = with_llm(context=context, llm=OpenAI(client=self.client))
        output = ""

        def sync_emit(blocks: List[Block], meta: Metadata):
            nonlocal output
            for block in blocks:
                if not block.is_text():
                    block.set_public_data(True)
                    output += f"({block.mime_type}: {block.raw_data_url})\n"
                    if block.mime_type == MimeTypes.PNG:
                        print("Image url for console:" +str(block.content_url))
                    if block.mime_type == MimeTypes.OGG_AUDIO:
                        print("audio url for console: " +str(block.content_url))                        
                else:
                    output += f"{block.text}\n"
        
        context.emit_funcs.append(sync_emit)
        self.run_agent(self._agent, context,msg_chat_id=context_id,use_dolly=flagged)

       
        return output
    
    @post("initial_index")
    def initial_index(self,chat_id:str =""):
        """Index a file from assets folder"""
        #logging.warning(str(self.usage.get_index_status(chat_id=chat_id)))

        if self.usage.get_index_status(chat_id=chat_id) == 0:

            filename="file.pdf"
            folder="assets/"    
            #handle, use letters and underscore
            file_handle = filename.replace(".","_")
            if not os.path.isfile(folder+filename):
                folder = "src/assets/"            
            try:
                with open(folder+filename,"rb") as f:             
                    bytes = f.read()
                    title_tag = Tag(kind=DocTag.TITLE, name=filename) 
                    source_tag = Tag(kind=DocTag.SOURCE, name=filename)
                    tags = [source_tag, title_tag]
                    pdf_file = File.create(self.client,content=bytes,mime_type=MimeTypes.PDF,tags=tags,handle=file_handle)                                             
                    pdf_file.set_public_data(True)
                    blockify_task = self.indexer_mixin.blockifier_mixin.blockify(pdf_file.id)
                    blockify_task.wait()
                    self.indexer_mixin.indexer_mixin.index_file(pdf_file.id)    
                    self.usage.set_index_status(chat_id=chat_id)
                    #logging.warning(str(self.usage.get_index_status(chat_id=chat_id)))
                
            except Exception as e:
                logging.warning(e)

        return "indexed"    
if __name__ == "__main__":

    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id=uuid.uuid4()
    
    print("chat id "+str(context_id))
    AgentREPL(MyAssistant,
            method="prompt",
           agent_package_config={'botToken': 'not-a-real-token-for-local-testing'       
        }).run_with_client(client=client,context_id=context_id ) 
    

    
    