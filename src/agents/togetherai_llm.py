from typing import List, Optional
import logging
from steamship import Block, File, PluginInstance, Steamship, Tag  #upm package(steamship)
from steamship.agents.schema import LLM, ChatLLM, Tool  #upm package(steamship)
from steamship.data import TagKind
from steamship.data.tags.tag_constants import GenerationTag

PLUGIN_HANDLE = "together-ai-generator"
DEFAULT_MAX_TOKENS = 256


class TogetherAiLLM(LLM):
    """LLM that uses Steamship's LLama plugin to generate completions.

    NOTE: By default, Valid model
    choices are
    """

    generator: PluginInstance
    client: Steamship

    def __init__(self,
                 client,
                 api_key: str = "",
                 model_name: str = "NousResearch/Nous-Hermes-Llama2-13b",
                 temperature: float = 0.4,
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
            options["stop"] = ["</s>", "\n\n", "<|", "\n###","<|im_end|>","<|im_start|>",f"\n\n{stop}"]
        else:
            options["stop"] = ["\n\n\n", "\n###","<|im_end|>","<|im_start|>","</s>"]

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "max_retries" in kwargs:
            options["max_retries"] = kwargs["max_retries"]
        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks


class ChatTogetherAiLLM(ChatLLM, TogetherAiLLM):
    """ChatLLM that uses Steamship's LLama plugin to generate chat completions."""

    def __init__(self,
                 client,
                 model_name: str = "NousResearch/Nous-Hermes-Llama2-13b",
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
        if len(messages) <= 0:
            return []
            
        #temp_file = File.create(client=self.client, blocks=messages)

        options = {}
        if len(tools) > 0:
            functions = []
            for tool in tools:
                functions.append(tool.as_openai_function())
            #options["functions"] = functions #disable functions
            #print(functions)
        
        options["stop"] = ["<|im_end|>","\n-","</s>","\nUser","\n\n\n"]
        
        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "max_retries" in kwargs:
            options["max_retries"] = kwargs["max_retries"]
            
        # for streaming use cases, we want to always use the existing file
        # the way to detect this would be if all messages were from the same file
        #disabled for now, always new file
        if self._from_same_existing_file(blocks=messages):
            file_id = messages[0].file_id
            block_indices = [b.index_in_file for b in messages]
            block_indices.sort()
            
            generate_task = self.generator.generate(
                input_file_id=file_id,
                input_file_block_index_list=block_indices,
                options=options,
                # append_output_to_file=True,  # not needed unless streaming. these can be ephemeral.
            )
            generate_task.wait()  # wait
            return generate_task.output.blocks

        # if not in same file, then we must create a temporary file and clean up after ourselves.
        try:
            tags = [Tag(kind=TagKind.GENERATION, name=GenerationTag.PROMPT_COMPLETION)]
            temp_file = File.create(client=self.client, blocks=messages, tags=tags)
            generate_task = self.generator.generate(input_file_id=temp_file.id, options=options)
            generate_task.wait()  # must wait until task is done before we can delete the file
            return generate_task.output.blocks
        finally:
            temp_file.delete()

    def _from_same_existing_file(self, blocks: List[Block]) -> bool:
        return False #always use new file for now
        if len(blocks) == 1:
            return blocks[0].file_id is not None
        file_id = blocks[0].file_id
        for b in blocks[1:]:
            if b.file_id != file_id:
                return False
        return True

