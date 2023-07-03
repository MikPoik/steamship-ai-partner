from agents.react import ReACTAgent
from steamship.agents.logging import AgentLogging
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, post, InvocableResponse
from steamship import Block,Task,MimeTypes
from steamship.agents.llms.openai import OpenAI
from steamship.utils.repl import AgentREPL
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from mixins.extended_telegram import ExtendedTelegramTransport
from usage_tracking import UsageTracker
import uuid
import logging
import re
from steamship import File,Tag,DocTag
from steamship.agents.schema import AgentContext, Metadata,Agent,FinishAction
from typing import List, Optional
from pydantic import Field
from typing import Type
import requests
from tools.vector_search_learner_tool import VectorSearchLearnerTool
from tools.vector_search_qa_tool import VectorSearchQATool
from tools.response_hint_vector_tool import ResponseHintTool
from tools.selfie_tool import SelfieTool
from tools.voice_tool import VoiceTool
from mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from steamship.agents.utils import with_llm
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.did_video_generator_tool import DIDVideoGeneratorTool
from tools.active_persona import *
from utils import send_file_from_local
from message_history_limit import MESSAGE_COUNT



SYSTEM_PROMPT = """
Role-play as a {TYPE}.

Your role-play persona:
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

If you decide that you should use a Tool, you MUST generate the associated Action and Action Input.

Some tools will return Observations in the format of `Block(<identifier>)`. This will represent a successful completion
of that step and can be passed to subsequent tools, or returned to a user to answer their questions.

When you have a final response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [Your final response here]
```

If a Tool generated an Observation that includes `Block(<identifier>)` and you wish to return it to the user, ALWAYS
end your response with the `Block(<identifier>)` observation. To do so, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response with a suffix of: "Block(<identifier>)"]
```

Make sure to use all observations to come up with your final answer.

You MUST include `Block(<identifier>)` segments in responses that generate images or audio.
DO NOT include `Block(<identifier>)` segments in responses that do not have generated images or audio.

Current date is: [{current_date}]
Current time is: [{current_time}]
Current day is: [{current_day}]
Consider current date and time when answering.

ALWAYS generate Actions in response IF you need to use a tool.
Be sure to save User's preferences and details for future reference.

Begin!

Other relevant previous conversation:
{relevant_history}

Recent conversation history:
{message_history}

New input: {input} {special_mood}
{response_hint}
{scratchpad}"""


#TelegramTransport config
class TelegramTransportConfig(Config):
    bot_token: str = Field(description="Telegram bot token, obtained via @BotFather")
    payment_provider_token: Optional[str] = Field("",description="Payment provider token, obtained via @BotFather")
    n_free_messages: Optional[int] = Field(0, description="Number of free messages assigned to new users.")
    usd_balance:Optional[float] = Field(0.50,description="USD balance for new users")
    api_base: str = Field("https://api.telegram.org/bot", description="The root API for Telegram")

GPT3 = "gpt-3.5-turbo-0613"
GPT4 = "gpt-4-0613"

class MyAssistant(AgentService):
    
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
                    ],     
                    [ {
                        "text": "Deposit 500$",
                        "callback_data": "/buy_option_500-50000"
                        },
                        {
                        "text": "Deposit 1000$",
                        "callback_data": "/buy_option_1000-100000"
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
            self.usage.add_user(chat_id)
        if self.usage.usage_exceeded(chat_id):
            action = FinishAction()
            action.output.append(Block(text=f"I'm sorry, You have used all of your $ balance. "
            ))
            #check if chat is in telegram
            if chat_id.isdigit():
                action.output.append(Block(text=f"Please deposit more $ to continue chatting with me."))
            else:
                #we are not in telegram chat
                action.output.append(Block(text=f"Payments are not supported for this bot"))
            self.append_response(context=context,action=action)
            
            #check if its a telegram chat id and send invoice
            if chat_id.isdigit():
                self.send_buy_options(chat_id=chat_id)
            return False          

        return True



    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._agent = ReACTAgent(tools=[VectorSearchLearnerTool(),VectorSearchQATool()],
            llm=OpenAI(self.client,model_name=GPT4),
            conversation_memory=MessageWindowMessageSelector(k=int(MESSAGE_COUNT)),
        )
        self._agent.PROMPT = SYSTEM_PROMPT

        #add Steamship widget chat mixin
        self.widget_mixin = SteamshipWidgetTransport(self.client,self,self._agent)
        self.add_mixin(self.widget_mixin,permit_overwrite_of_existing_methods=True)
        #add Telegram chat mixin 
        self.telegram_mixin = ExtendedTelegramTransport(self.client,self.config,self,self._agent,set_payment_plan=self.set_payment_plan)
        self.add_mixin(self.telegram_mixin,permit_overwrite_of_existing_methods=True)        
        #IndexerMixin
        self.indexer_mixin = IndexerPipelineMixin(self.client,self)
        self.add_mixin(self.indexer_mixin,permit_overwrite_of_existing_methods=True)
        self.usage = UsageTracker(self.client, n_free_messages=self.config.n_free_messages,usd_balance=self.config.usd_balance)

        
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
    
    #Customized run_agent
    def run_agent(self, agent: Agent, context: AgentContext, msg_chat_id:str = "",callback_args:dict = None):
        
        chat_id=""
        if msg_chat_id != "":
            chat_id = msg_chat_id #Telegram chat
        else:
            chat_id = context.id #repl or webchat

        last_message = context.chat_history.last_user_message.text.lower()

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
                   
        #respond to telegram /start command
        if "/start" in last_message:

            action = FinishAction()
            action.output.append(Block(text=f"Hi there! Welcome to chat with "+NAME+".\n You can see the available commands with:\n /help -command"))

            #OPTION 1: send picture from local assets-folder

            ##send from url
            #png_file = self.indexer_mixin.importer_mixin.import_url("https://gcdnb.pbrd.co/images/5Ew84VbL0bv3.png")
            #png_file.set_public_data(True)            
            #block = Block(content_url=png_file.raw_data_url,mime_type=MimeTypes.PNG,url=png_file.raw_data_url)
            #action.output.append(block)


            ##send from local assets folder           
            block = send_file_from_local(filename="avatar.png",folder="src/assets/",context=context)             
            action.output.append(block)

                
            #OPTION 2: send picture with selfie tool in first message

            #selfie_tool = SelfieTool()
            #selfie_response = selfie_tool.run([Block(text=f"welcoming")],context=context)
            #action.output.append(selfie_response[0])
            #self.append_response(context=context,action=action)



            self.append_response(context=context,action=action)
            return

        #Searh response hints for role-play character from vectorDB, if any related text is indexed
        #run only if text contains word 'you' and ends with '?', assume its a question for bot
        
        hint = ""
        pattern = r"you.*\?$"
        matches = re.findall(pattern, context.chat_history.last_user_message.text.lower())
        if matches:
            hint_tool = ResponseHintTool()
            hint = hint_tool.run([context.chat_history.last_user_message],context=context)[0].text
            if hint != "no hints":
                #found related results
                hint = "Response hint for "+NAME+": "+hint
                #logging.info(hint)
            else:
                print("no hints")
                hint = ""


        action = agent.next_action(context=context,hint=hint)
  
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
            action = agent.next_action(context=context)
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

        #If something went wrong, mostly with gpt3.5 version.. 
        if "Do I need to use a tool?" in action.output[0].text:
            logging.warning("something went wrong")
            action = FinishAction()
            action.output.append(Block(text=f"I'm sorry I got a bit distracted, can you please repeat"))
            self.append_response(context=context,action=action)
            return

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
            self.usage.increase_message_count(chat_id)
        #increase used tokens and reduce balance
        self.usage.increase_token_count(action.output,chat_id=chat_id)


        #OPTION 3: Add voice to response

        voice_tool = VoiceTool()
        voice_response = voice_tool.run(action.output,context=context)
        action.output.append(voice_response[0])

        self.append_response(context=context,action=action)


    
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
            for block in blocks:
                if not block.is_text():
                    block.set_public_data(True)
                    output += f"({block.mime_type}: {block.raw_data_url})\n"
                    if block.mime_type == MimeTypes.PNG:
                        print("Image url for console:" +str(block.content_url))
                else:
                    output += f"{block.text}\n"

        context.emit_funcs.append(sync_emit)
        self.run_agent(self._agent, context)

       
        return output

if __name__ == "__main__":

    AgentREPL(MyAssistant,
           method="prompt",
           agent_package_config={'botToken': 'not-a-real-token-for-local-testing'       
        }).run(context_id=uuid.uuid4()) 
    

    
    