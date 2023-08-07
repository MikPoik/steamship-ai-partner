from typing import List, Optional

from steamship import Block, Steamship, PluginInstance
from steamship.agents.schema import LLM

PLUGIN_HANDLE = "dolly-lm-test"
REPLICATE_API_KEY = ""
DEFAULT_MAX_TOKENS = 256

class DollyLLM(LLM):
    generator: PluginInstance
    client: Steamship
    _max_tokens: int
    _temperature: float
    _api_key:str
    

    def __init__(
        self, client, max_tokens: int = 500, temperature: float = 0.2,api_key:str="", *args, **kwargs
    ):
        client=client
        if "max_tokens" in kwargs:
          self._max_tokens = kwargs["max_tokens"]
        if "temperature" in kwargs:
            self._temperature = kwargs["temperature"]
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._api_key = api_key

        generator = client.use_plugin(PLUGIN_HANDLE,
                                      config={"replicate_api_key" : self._api_key})


        super().__init__(client=client, generator=generator, *args, **kwargs)

    def complete(self, prompt: str, stop: Optional[str] = None,**kwargs) -> List[Block]:
        
        options = {"max_tokens":self._max_tokens,
                   "temperature":self._temperature
                   }

        if stop:
            options["stop"] = stop

        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            options["temperature"] = kwargs["max_tokens"]
            
        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks