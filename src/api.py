from agents.functions_based import FunctionsBasedAgent  #upm package(steamship)
from steamship.agents.logging import AgentLogging  #upm package(steamship)
from steamship.agents.service.agent_service import AgentService  #upm package(steamship)
from steamship.invocable import Config, post  #upm package(steamship)
from steamship import Block, Task, MimeTypes, Steamship, SteamshipError  #upm package(steamship)
from agents.openai import ChatOpenAI, OpenAI  #upm package(steamship)
from steamship.utils.repl import AgentREPL  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport  #upm package(steamship)
from mixins.extended_telegram import ExtendedTelegramTransport, TelegramTransportConfig
from usage_tracking import UsageTracker  #upm package(steamship)
import uuid, os, re, logging, requests
from steamship import File, Tag, DocTag  #upm package(steamship)
from steamship.agents.schema import AgentContext, Metadata, Agent, FinishAction, EmitFunc  #upm package(steamship)
from typing import List, Optional
from pydantic import Field
from typing import Type
from collections import defaultdict
from tools.selfie_tool_kandinsky import SelfieToolKandinsky  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
from tools.voice_tool_ogg import VoiceToolOGG  #upm package(steamship)
from tools.voice_tool_mp3 import VoiceToolMP3  #upm package(steamship)
from steamship.invocable.mixins.blockifier_mixin import BlockifierMixin  #upm package(steamship)
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin  #upm package(steamship)
from steamship.invocable.mixins.indexer_mixin import IndexerMixin  #upm package(steamship)
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin  #upm package(steamship)
from tools.active_companion import *  #upm package(steamship)
from utils import send_file_from_local  #upm package(steamship)
from agents.togetherai_llm import ChatTogetherAiLLM  #upm package(steamship)
from agents.react import ReACTAgent  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from tools.lemonfox_tts_tool import LemonfoxTTSTool  #upm package(steamship)
from agents.zephyr_llm import ChatZephyr, Zephyr  #upm package(steamship)
from tools.selfie_tool_fal_ai import SelfieToolFalAi  #upm package(steamship)
import os
from json import loads

#Available llm models to use
GPT3 = "gpt-3.5-turbo-0613"
GPT4 = "gpt-4-0613"
MISTRAL = "teknium/OpenHermes-2-Mistral-7B"
MISTRAL25 = "teknium/OpenHermes-2p5-Mistral-7B"
ZEPHYR_CHAT = "zephyr-chat"
MYTHOMAX = "Gryphe/MythoMax-L2-13b"
MIXTRAL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
SFT_MIXTRAL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
YI34B = "NousResearch/Nous-Hermes-2-Yi-34B"

os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"


#TelegramTransport config
class MyAssistantConfig(Config):
    api_base: Optional[str] = Field("https://api.telegram.org/bot",
                                    description="The root API for Telegram")
    bot_token: str = Field(
        "", description="Telegram bot token, obtained via @BotFather")
    payment_provider_token: Optional[str] = Field(
        ":TEST:",
        description="Payment provider token, obtained via @BotFather")
    transloadit_api_key: str = Field(
        "", description="Transloadit.com api key for OGG encoding")
    transloadit_api_secret: str = Field(
        "", description="Transloadit.com api secret")
    use_voice: str = Field(
        "none",
        description=
        "Send voice messages addition to text, values: ogg, mp3,coqui or none")
    llm_model: Optional[str] = Field(ZEPHYR_CHAT,
                                     description="llm model to use")
    together_ai_api_key: Optional[str] = Field(
        "",
        description="Together.ai api key")

    zephyr_api_key: Optional[str] = Field("",
                                          description="Lemonfox api key")
    create_images: Optional[str] = Field(
        "true", description="Enable Image generation tool")

    image_model: Optional[str] = Field(
        "realistic-vision-v3",
        description="CivitAI URL or getimg.ai model name, for cli testing")
    verbose_logging: Optional[bool] = Field(
        False, description="Enable verbose cli logging")


def build_context_appending_emit_func(
        context: AgentContext,
        make_blocks_public: Optional[bool] = False) -> EmitFunc:
    """Build an emit function that will append output blocks directly to ChatHistory, via AgentContext.
  
  NOTE: Messages will be tagged as ASSISTANT messages, as this assumes that agent output should be considered
  an assistant response to a USER.
  """

    def chat_history_append_func(blocks: List[Block], metadata: Metadata):
        for block in blocks:
            block.set_public_data(make_blocks_public)
            try:

                context.chat_history.append_assistant_message(
                    text=block.text,
                    tags=block.tags,
                    url=block.raw_data_url or block.url or block.content_url
                    or None,
                    mime_type=block.mime_type,
                )
            except Exception as e:
                logging.warning(e)
                logging.warning("failed to save assistant message")

    return chat_history_append_func


class MyAssistant(AgentService):

    USED_MIXIN_CLASSES = [
        IndexerPipelineMixin, FileImporterMixin, BlockifierMixin, IndexerMixin,
        ExtendedTelegramTransport
        #, SteamshipWidgetTransport  #Uncomment to enable webwidget support
    ]

    config: MyAssistantConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return MyAssistantConfig

    def append_response(self, context: AgentContext, action: FinishAction):
        for func in context.emit_funcs:
            func(action.output, context.metadata)

    def check_usage(self, chat_id: str, context: AgentContext) -> bool:

        if not self.usage.exists(chat_id):
            self.usage.add_user(str(chat_id))

        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        tools = []
        if "true" in self.config.create_images:
            tools = [SelfieTool(), SelfieToolFalAi()]

        if "gpt" in self.config.llm_model:
            self.set_default_agent(
                FunctionsBasedAgent(
                    tools,
                    llm=ChatOpenAI(self.client,
                                   model_name=self.config.llm_model,
                                   temperature=0.8,
                                   max_tokens=256,
                                   moderate_output=False),
                    client=self.client,
                    message_selector=MessageWindowMessageSelector(
                        k=MESSAGE_COUNT)))

        if not "zephyr-chat" in self.config.llm_model and "gpt" not in self.config.llm_model:
            self.set_default_agent(
                FunctionsBasedAgent(
                    tools,
                    llm=ChatTogetherAiLLM(
                        self.client,
                        api_key=self.config.together_ai_api_key,
                        model_name=self.config.llm_model,
                        temperature=0.8,
                        #top_p=0.7,
                        max_tokens=256,
                        max_retries=4),
                    client=self.client,
                    message_selector=MessageWindowMessageSelector(
                        k=MESSAGE_COUNT)))

        if "zephyr-chat" in self.config.llm_model:
            self.set_default_agent(
                FunctionsBasedAgent(
                    tools,
                    llm=ChatZephyr(
                        self.client,
                        api_key=self.config.zephyr_api_key,
                        model_name=self.config.llm_model,
                        temperature=0.8,
                        #top_p=0.7,
                        max_tokens=256,
                        max_retries=4),
                    client=self.client,
                    message_selector=MessageWindowMessageSelector(
                        k=MESSAGE_COUNT)))

        # This Mixin provides HTTP endpoints that connects this agent to a web client
        # Uncomment to enable webwidget chat
        #self.add_mixin(
        #    SteamshipWidgetTransport(client=self.client, agent_service=self))

        #IndexerMixin
        self.indexer_mixin = IndexerPipelineMixin(self.client, self)
        self.add_mixin(self.indexer_mixin)

        #self.usage = UsageTracker(self.client, n_free_messages=1)

    #Customized run_agent
    def run_agent(self,
                  agent: Agent,
                  context: AgentContext,
                  msg_chat_id: str = "",
                  callback_args: dict = None):
        #self.usage.set_kv_store(self.client, storage_identifier=context.id)
        context.completed_steps = []

        #Check used messages, if exceeded, send message and invoice (invoice only in telegram)
        #Disabled usage check, slows down the bot
        #if not self.check_usage(chat_id=chat_id, context=context):
        #    return

        action = self.next_action(
            agent=agent,
            input_blocks=[context.chat_history.last_user_message],
            context=context)

        # Set the counter for the number of actions run.
        # This enables the agent to enforce a budget on actions to guard against running forever.
        number_of_actions_run = 0
        actions_per_tool = defaultdict(lambda: 0)

        while not action.is_final:
            # If we've exceeded our Action Budget, throw an error.
            if number_of_actions_run >= self.max_actions_per_run:
                raise SteamshipError(message=(
                    f"Agent reached its Action budget of {self.max_actions_per_run} without arriving at a response. If you are the developer, checking the logs may reveal it was selecting unhelpful tools or receiving unhelpful responses from them."
                ))

            if action.tool and action.tool in self.max_actions_per_tool:
                if actions_per_tool[action.tool] > self.max_actions_per_tool[
                        action.tool]:
                    raise SteamshipError(message=(
                        f"Agent reached its Action budget of {self.max_actions_per_tool[action.tool]} for tool {action.tool} without arriving at a response. If you are the developer, checking the logs may reveal it was selecting unhelpful tools or receiving unhelpful responses from them."
                    ))
            # Run the next action and increment our counter
            self.run_action(agent=agent, action=action, context=context)
            number_of_actions_run += 1
            if action.tool:
                actions_per_tool[action.tool] += 1
            if action.is_final:
                break
            action = self.next_action(agent=agent,
                                      input_blocks=action.output,
                                      context=context)
            # TODO: Arrive at a solid design for the details of this structured log object
            logging.info(
                f"Next Tool: {action.tool}",
                extra={
                    AgentLogging.TOOL_NAME: action.tool,
                    AgentLogging.IS_MESSAGE: False,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.ACTION,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                },
            )
        agent.record_action_run(action, context)

        output_text_length = 0
        if action.output is not None:
            output_text_length = sum(
                [len(block.text or "") for block in action.output])
        logging.info(
            f"Completed agent run. Result: {len(action.output or [])} blocks. {output_text_length} total text length. Emitting on {len(context.emit_funcs)} functions."
        )

        current_voice_config = self.config.use_voice
        meta_voice_id = context.metadata.get("instruction", {}).get("voice_id")
        if meta_voice_id is not None:
            if meta_voice_id != "none":
                #logging.warning("coqui voiceid: " + meta_voice_id)
                current_voice_config = meta_voice_id
            elif meta_voice_id == "none":
                current_voice_config = "none"

        voice_response = []
        #OPTION 3: Add voice to response
        if "ogg" in current_voice_config:
            voice_tool = VoiceToolOGG()
            voice_response = voice_tool.run(
                action.output,
                context=context,
                transloadit_api_key=self.config.transloadit_api_key,
                transloadit_api_secret=self.config.transloadit_api_secret)
            action.output.append(voice_response[0])
        elif "mp3" in current_voice_config:  ## if default audio format (change voice_tool_orig.py to voice_tool.py):
            voice_tool = VoiceToolMP3()
            voice_response = voice_tool.run(action.output, context=context)
            action.output.append(voice_response[0])
        elif current_voice_config != "none":
            voice_tool = LemonfoxTTSTool()
            voice_response = voice_tool.run(
                action.output,
                context=context,
            )
            for block in voice_response:
                action.output.append(block)

        self.append_response(context=context, action=action)

    @post("clear_history")
    def clear_history(self, context_id: Optional[str] = None):
        context = self.build_default_context(context_id)
        context.chat_history.clear()
        return "OK"

    @post("append_history")
    def append_history(self,
                       prompt: Optional[str] = None,
                       context_id: Optional[str] = None):
        logging.warning(prompt)
        context = self.build_default_context(context_id)

        if prompt:
            try:
                # Parse the JSON string to extract messages
                messages = loads(prompt)
                context = self.build_default_context(context_id)

                # Loop through each message and add it to chat history
                for message in messages:
                    if message['role'] == 'assistant':
                        context.chat_history.append_assistant_message(
                            message['content'])
                    elif message['role'] == 'user':
                        context.chat_history.append_user_message(
                            message['content'])
            except Exception as e:
                logging.warning(
                    "Failed to parse prompt or append to chat history: " +
                    str(e))
        return "OK"

    @post("prompt")
    def prompt(self,
               prompt: Optional[str] = None,
               context_id: Optional[str] = None,
               name: Optional[str] = None,
               personality: Optional[str] = None,
               description: Optional[str] = None,
               behaviour: Optional[str] = None,
               selfie_pre: Optional[str] = None,
               selfie_post: Optional[str] = None,
               seed: Optional[str] = None,
               model: Optional[str] = None,
               image_model: Optional[str] = None,
               voice_id: Optional[str] = None,
               create_images: Optional[str] = None,
               is_pro: Optional[str] = None,
               **kwargs) -> List[Block]:
        """Run an agent with the provided text as the input."""
        #print("context_id: "+context_id)
        with self.build_default_context(context_id, **kwargs) as context:
            prompt = prompt or kwargs.get("question")

            context.metadata["verbose_logging"] = self.config.verbose_logging

            context.metadata["instruction"] = {
                "name": name or NAME,
                "personality": personality or PERSONA,
                "type": description or TYPE,
                "behaviour": behaviour or BEHAVIOUR,
                "selfie_pre": selfie_pre or NSFW_SELFIE_TEMPLATE_PRE,
                "selfie_post": selfie_post or NSFW_SELFIE_TEMPLATE_POST,
                "seed": seed or SEED,
                "model": model or self.config.llm_model,
                "image_model": image_model or self.config.image_model,
                "voice_id": voice_id or self.config.use_voice,
                "create_images": create_images or self.config.create_images,
                "is_pro": is_pro or None,
                "context_id": context_id
            }

            last_agent_msg = context.chat_history.last_agent_message

            if not last_agent_msg:
                meta_seed = context.metadata.get("instruction", {}).get("seed")
                if meta_seed is not None:
                    context.chat_history.append_assistant_message(
                        f'{meta_seed}')

            context.chat_history.append_user_message(f"{prompt}")

            #logging.warning("prompt inputs: " +
            #str(context.metadata["instruction"]))
            output_blocks = []

            def sync_emit(blocks: List[Block], meta: Metadata):
                nonlocal output_blocks
                output_blocks.extend(blocks)

            context.emit_funcs.append(sync_emit)
            context.emit_funcs.append(
                build_context_appending_emit_func(context=context,
                                                  make_blocks_public=True))
            # Get the agent

            agent: Optional[Agent] = self.get_default_agent()
            self.run_agent(agent, context)

            # Return the response as a set of multi-modal blocks.
            return output_blocks

    @post("generate_avatar", public=True)
    def generate_avatar(self,
                        prompt: Optional[str] = None,
                        context_id: Optional[str] = None,
                        name: Optional[str] = None,
                        personality: Optional[str] = None,
                        description: Optional[str] = None,
                        behaviour: Optional[str] = None,
                        selfie_pre: Optional[str] = None,
                        selfie_post: Optional[str] = None,
                        seed: Optional[str] = None,
                        model: Optional[str] = None,
                        image_model: Optional[str] = None,
                        voice_id: Optional[str] = None,
                        create_images: Optional[str] = None,
                        chat_id: str = ""):
        """Run an agent with the provided text as the input."""

        with self.build_default_context(context_id=chat_id) as context:
            context.metadata["instruction"] = {
                "name": name or None,
                "personality": personality or None,
                "type": description or None,
                "behaviour": behaviour or None,
                "selfie_pre": selfie_pre or None,
                "selfie_post": selfie_post or None,
                "seed": seed or None,
                "model": model or None,
                "image_model": image_model or None,
                "voice_id": voice_id or None,
                "create_images": self.config.create_images or None
            }

            # Get the agent
            agent: Optional[Agent] = self.get_default_agent()
            get_img_ai_models = [
                "realistic-vision-v3", "dark-sushi-mix-v2-25",
                "absolute-reality-v1-8-1", "van-gogh-diffusion",
                "neverending-dream", "mo-di-diffusion", "synthwave-punk-v2",
                "dream-shaper-v8", "arcane-diffusion"
            ]
            if image_model is not None and image_model not in get_img_ai_models:
                selfie_tool = SelfieToolFalAi()
                selfie_response = selfie_tool.run([Block(text=prompt)],
                                                  context=context,
                                                  img_height=512,
                                                  img_width=512,
                                                  stream=False)
                return selfie_response
            else:
                selfie_tool = SelfieTool()
                selfie_response = selfie_tool.run([Block(text=prompt)],
                                                  context=context,
                                                  img_height=512,
                                                  img_width=512,
                                                  stream=False)

                return selfie_response

    @post("initial_index")
    def initial_index(self, chat_id: str = ""):
        """Index a file from assets folder"""
        #logging.warning(str(self.usage.get_index_status(chat_id=chat_id)))

        if self.usage.get_index_status(chat_id=chat_id) == 0:

            filename = "file.pdf"
            folder = "assets/"
            #handle, use letters and underscore
            file_handle = filename.replace(".", "_")
            if not os.path.isfile(folder + filename):
                folder = "src/assets/"
            try:
                with open(folder + filename, "rb") as f:
                    bytes = f.read()
                    title_tag = Tag(kind=DocTag.TITLE, name=filename)
                    source_tag = Tag(kind=DocTag.SOURCE, name=filename)
                    tags = [source_tag, title_tag]
                    pdf_file = File.create(self.client,
                                           content=bytes,
                                           mime_type=MimeTypes.PDF,
                                           tags=tags,
                                           handle=file_handle)
                    pdf_file.set_public_data(True)
                    blockify_task = self.indexer_mixin.blockifier_mixin.blockify(
                        pdf_file.id)
                    blockify_task.wait()
                    self.indexer_mixin.indexer_mixin.index_file(pdf_file.id)
                    self.usage.set_index_status(chat_id=chat_id)
                    #logging.warning(str(self.usage.get_index_status(chat_id=chat_id)))

            except Exception as e:
                logging.warning(e)

        return "indexed"


if __name__ == "__main__":
    #your workspace name
    client = Steamship(workspace="partner-ai-dev4-ws")
    context_id = uuid.uuid4()
    #context_id="mipotest2"
    #print("chat id " + str(context_id))
    AgentREPL(MyAssistant,
              method="prompt",
              agent_package_config={
                  'botToken': 'not-a-real-token-for-local-testing'
              },
              context_id=context_id).run_with_client(client=client)
