from typing import List, Optional

from steamship import Block, Steamship, PluginInstance #upm package(steamship)
from steamship.agents.schema import LLM #upm package(steamship)

PLUGIN_HANDLE = "llama-hermes-small-test"
GW_API_KEY = ""
DEFAULT_MAX_TOKENS = 256

class LlamaGWLLM(LLM):
    generator: PluginInstance
    client: Steamship
    _max_tokens: int
    _temperature: float
    _api_key:str
    _model_name:str
    

    def __init__(
        self, client,api_key:str="", model_name: str = "NousResearch/Nous-Hermes-Llama2-7b", max_tokens: int = 256, temperature: float = 0.4, *args, **kwargs
    ):
        client=client
        if "max_tokens" in kwargs:
          self._max_tokens = kwargs["max_tokens"]
        if "temperature" in kwargs:
            self._temperature = kwargs["temperature"]
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._api_key = api_key
        self._model_name = model_name

        generator = client.use_plugin(PLUGIN_HANDLE,
                                      config={"api_key" : self._api_key,"model":self._model_name})


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
        if "presence_penalty" in kwargs:
            options["presence_penalty"] = kwargs["presence_penalty"]
        if "presence_penalty" in kwargs:
            options["presence_penalty"] = kwargs["presence_penalty"]            

        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks