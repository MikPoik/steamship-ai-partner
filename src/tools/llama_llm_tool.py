"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
import json
from tools.vector_search_response_tool import VectorSearchResponseTool
import requests
from tools.active_persona import *
from message_history_limit import *
import logging
from tools.active_persona import *


#Pre-prompt/prompt format
PROMPT_FORMAT = """### human: Act as a girlfriend to male. You do as requested and are willing to pleasure. Do not attach links. Your response word limit is up to 256 words.

### response: Sure! sounds good.

### human: {input}

### response:"""


class LlamaLLMTool(Tool):
    """Tool to generat text using Llama2 llm"""

    rewrite_prompt = PROMPT_FORMAT

    name: str = "Llama2LLMTool"
    human_description: str = "Generates text with uncensored Llama2 llm."
    agent_description = (
        "Used to generate text with Llama llm.  "
        "The input is the text that you want to say. "
        "The output is the text generated."
    )
    generator_plugin_handle: str = "llama-llm"
    generator_plugin_config: dict = {}


    def run(self, tool_input: List[Block], context: AgentContext, context_id:str = "",vector_response:str="",api_url = "",api_key="") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        headers = {
            "x-api-key" :api_key,
            "Content-Type": "application/json"}        
        
        #TODO add chat history to prompt
        prompt = self.rewrite_prompt.format(input=tool_input[0].text)    
        #print(prompt)
        json = {"prompt": prompt,        
                "temperature":0.4,
                "max_tokens": 256}
        response = requests.post(api_url, headers=headers,json=json)
        result_text =""
        result_text = response.json()


        if response.status_code == 200:
            #should we clean output here, if puffin creates more responses..
            result_text = response.json()
            context.chat_history.append_assistant_message(text=result_text)
        else:
            print("Request failed with status code:", response.status_code)
            result_text = response.status_code
    

        return [Block(text=result_text)]

        


if __name__ == "__main__":
    tool = LlamaLLMTool()
    client = Steamship(workspace="partner-ai-dev3-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
