from typing import List, Optional

from steamship import Block, File, PluginInstance, Steamship  #upm package(steamship)
from steamship.agents.schema import LLM, ChatLLM, Tool  #upm package(steamship)

PLUGIN_HANDLE = "lemonfox-streaming-llm"
DEFAULT_MAX_TOKENS = 256


class Zephyr(LLM):
    """LLM that uses Steamship's LLama plugin to generate completions.

    NOTE: By default, Valid model
    choices are
    """

    generator: PluginInstance
    client: Steamship

    def __init__(self,
                 client,
                 api_key: str = "",
                 model_name: str = "zephyr-chat",
                 temperature: float = 0.9,
                 *args,
                 **kwargs):
        """Create a new instance.

        Valid model names are:


        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        client = client
        max_tokens = DEFAULT_MAX_TOKENS
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]

        generator = client.use_plugin(
            PLUGIN_HANDLE,
            version="1.0.1",
            config={
                "api_key": api_key,
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
        )
        super().__init__(client=client, generator=generator, *args, **kwargs)

    def complete(self,
                 prompt: str,
                 stop: Optional[str] = None,
                 **kwargs) -> List[Block]:
        """Completes the prompt, respecting the supplied stop sequence.

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        options = {}
        if stop:
            stop = stop.split(" ")[0]
            options["stop"] = ["\n\n", "\n###", "\nHuman", f"\n{stop}"]
        else:
            options["stop"] = ["\n\n\n", "\n###", "\nHuman", "<|"]
        #print(options)

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "max_retries" in kwargs:
            options["max_retries"] = kwargs["max_retries"]
        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks


class ChatZephyr(ChatLLM, Zephyr):
    """ChatLLM that uses Steamship's LLama plugin to generate chat completions."""

    def __init__(self,
                 client,
                 model_name: str = "zephyr-chat",
                 *args,
                 **kwargs):
        """Create a new instance.

        Valid model names are:


        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """
        super().__init__(client=client, model_name=model_name, *args, **kwargs)

    def chat(self, messages: List[Block], tools: Optional[List[Tool]],
             **kwargs) -> List[Block]:
        """Sends chat messages to the LLM with functions from the supplied tools in a side-channel.

        Supported kwargs include:
        - `max_tokens` (controls the size of LLM responses)
        """

        temp_file = File.create(client=self.client, blocks=messages)

        options = {}
        if len(tools) > 0:
            functions = []
            for tool in tools:
                functions.append(tool.as_openai_function())
            options["functions"] = functions
            #print(functions)

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "max_retries" in kwargs:
            options["max_retries"] = kwargs["max_retries"]

        tool_selection_task = self.generator.generate(
            input_file_id=temp_file.id, options=options)
        tool_selection_task.wait()
        return tool_selection_task.output.blocks
