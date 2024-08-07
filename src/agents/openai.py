import json
import logging
from typing import List, Optional

from steamship import Block, File, PluginInstance, Steamship, Tag
from steamship.agents.logging import AgentLogging
from steamship.agents.schema import LLM, ChatLLM, Tool
from steamship.data import TagKind
from steamship.data.tags.tag_constants import GenerationTag
from steamship.cli.utils import is_in_replit
PLUGIN_HANDLE = "gpt-4"
DEFAULT_MAX_TOKENS = 256


class OpenAI(LLM):
    """LLM that uses Steamship's OpenAI plugin to generate completions.

    NOTE: By default, this LLM uses the `gpt-3.5-turbo` model. Valid model
    choices are `gpt-3.5-turbo` and `gpt-4`.
    """

    generator: PluginInstance
    client: Steamship

    def __init__(
        self, client, model_name: str = "gpt-3.5-turbo", temperature: float = 0.4,top_p:float = 1,frequency_penalty:float = 0,presence_penalty = 0, *args, **kwargs
    ):
        """Create a new instance.

        Valid model names are:
         - gpt-4
         - gpt-3.5-turbo

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        client = client
        max_tokens = DEFAULT_MAX_TOKENS
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]

        generator = client.use_plugin(
            PLUGIN_HANDLE,
            config={"model": model_name, "temperature": temperature, "max_tokens": max_tokens,"top_p":top_p,"frequency_penalty":frequency_penalty,"presence_penalty":presence_penalty},
        )
        super().__init__(client=client, generator=generator, *args, **kwargs)

    def complete(self, prompt: str, stop: Optional[str] = None, **kwargs) -> List[Block]:
        """Completes the prompt, respecting the supplied stop sequence.

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        options = {}
        if stop:
            options["stop"] = "\n\nHuman" #override stop word

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]

        # TODO(dougreid): do we care about streaming here? should we take a kwarg that is file_id ?
        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks


class ChatOpenAI(ChatLLM, OpenAI):
    """ChatLLM that uses Steamship's OpenAI plugin to generate chat completions."""

    def __init__(self, client, model_name: str = "gpt-4-0613", *args, **kwargs):
        """Create a new instance.

        Valid model names are:
         - gpt-4
         - gpt-4-0613

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        super().__init__(client=client, model_name=model_name, *args, **kwargs)

    def chat(self, messages: List[Block], tools: Optional[List[Tool]], **kwargs) -> List[Block]:
        """Sends chat messages to the LLM with functions from the supplied tools in a side-channel.

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        if len(messages) <= 0:
            return []

        options = {}
        if len(tools) > 0:
            functions = []
            for tool in tools:
                functions.append(tool.as_openai_function().dict())
            #options["functions"] = functions #disable functions

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]

        extra = {
            AgentLogging.LLM_NAME: "OpenAI",
            AgentLogging.IS_MESSAGE: True,
            AgentLogging.MESSAGE_TYPE: AgentLogging.PROMPT,
            AgentLogging.MESSAGE_AUTHOR: AgentLogging.LLM,
        }

        if logging.WARNING >= logging.root.getEffectiveLevel():
            extra["messages"] = json.dumps(
                "\n".join([f"[{msg.chat_role}] {msg.as_llm_input()}" for msg in messages])
            )
            extra["tools"] = ",".join([t.name for t in tools])
        else:
            extra["num_messages"] = len(messages)
            extra["num_tools"] = len(tools)

        #logging.info(f"OpenAI ChatComplete ({messages[-1].as_llm_input()})", extra=extra)

        # for streaming use cases, we want to always use the existing file
        # the way to detect this would be if all messages were from the same file
        #stream = not is_in_replit()
        stream = False
        if self._from_same_existing_file(blocks=messages): 
            file_id = messages[0].file_id
            block_indices = [b.index_in_file for b in messages]
            block_indices.sort()
            logging.debug(f"OpenAI ChatComplete block_indices [{block_indices}]")
            generate_task = self.generator.generate(
                input_file_id=file_id,
                input_file_block_index_list=block_indices,
                options=options,
                streaming=stream,
                append_output_to_file=True,  # not needed unless streaming. these can be ephemeral.
            )
            generate_task.wait()  # wait
            return generate_task.output.blocks

        # if not in same file, then we must create a temporary file and clean up after ourselves.
        try:
            tags = [Tag(kind=TagKind.GENERATION, name=GenerationTag.PROMPT_COMPLETION)]
            temp_file = File.create(client=self.client, blocks=messages, tags=tags)
            generate_task = self.generator.generate(input_file_id=temp_file.id, options=options,streaming=stream,append_output_to_file=True)
            generate_task.wait()  # must wait until task is done before we can delete the file
            return generate_task.output.blocks
        finally:
            temp_file.delete()

    def _from_same_existing_file(self, blocks: List[Block]) -> bool:
        return False #return False always otherwise system prompt will be inserted in wrong position. Should be change prompt dynamic prompt to init and kv store to store the prompt.
        if len(blocks) == 1:
            return blocks[0].file_id is not None
        file_id = blocks[0].file_id
        for b in blocks[1:]:
            if b.file_id != file_id:
                return False
        return True
