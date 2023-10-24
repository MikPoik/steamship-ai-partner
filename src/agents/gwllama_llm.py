from typing import List, Optional

from steamship import Block, Steamship, PluginInstance  #upm package(steamship)
from steamship.agents.schema import LLM  #upm package(steamship)

PLUGIN_HANDLE = "llama-hermes-small-test"
DEFAULT_MAX_TOKENS = 256


class LlamaGWLLM(LLM):
  generator: PluginInstance
  client: Steamship

  def __init__(self,
               client,
               api_key: str = "",
               model_name: str = "NousResearch/Nous-Hermes-Llama2-7b",
               max_tokens: int = 256,
               temperature: float = 0.4,
               presence_penalty: float = 1.18,
               top_p: float = 1,
               *args,
               **kwargs):

    client = client

    generator = client.use_plugin(PLUGIN_HANDLE,
                                  config={
                                      "api_key": api_key,
                                      "model": model_name,
                                      "temperature": temperature,
                                      "max_tokens": max_tokens,
                                      "presence_penalty": presence_penalty,
                                      "top_p": top_p
                                  })

    super().__init__(client=client, generator=generator, *args, **kwargs)

  def complete(self,
               prompt: str,
               stop: Optional[str] = None,
               **kwargs) -> List[Block]:

    options = {}

    if stop:
      options["stop"] = stop

    #print(options)
    action_task = self.generator.generate(text=prompt, options=options)
    action_task.wait()
    return action_task.output.blocks
