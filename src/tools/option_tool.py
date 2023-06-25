from steamship import Steamship
from steamship.agents.llms import OpenAI
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from typing import Any, List, Union
from steamship import Block, Task
from steamship.agents.schema import AgentContext, Tool
from steamship.agents.utils import get_llm, with_llm
from steamship.utils.kv_store import KeyValueStore
import time
import json


DEFAULT_PROMPT = """I will reply to you with: {mode}"""

#KV key
OPTION_KEY = "agent-option"

class OptionTool(Tool):
    """
    Tool for generating response mood, prompt should be added in reACTagent prompt template after "new input"
    """

    name: str = "OptionTool"
    human_description: str = "Generate a response mood"
    agent_description: str = (
        "Use this tool to change prompt template"
        "The input is the option value"
        "The output is the result."
        

    )
    rewrite_prompt: str = DEFAULT_PROMPT


    def set_mode(self,mode: str, context: AgentContext):
        """Set a mode on the agent."""
        kv = KeyValueStore(context.client, OPTION_KEY)
        mode_settings = {
            "current_option": mode
        }
        # Note: The root value to the key-value store MUST be dict objects.
        kv.set("mode", mode_settings)
       
    def get_mode(self,context: AgentContext) -> str:
        """Get the mood on the agent."""
        kv = KeyValueStore(context.client,OPTION_KEY)

        option_settings = kv.get("mode") or {}
        current_option= option_settings.get("current_option", "text")  # Fails back to 'text' 
        return current_option 

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:

        # Extract mood keywords and values

        #get mood values from storage
        mode = self.get_mode(context=context)
        #print(mode)

        for block in tool_input:
        # If the block is not text, simply pass it through.
            if not block.is_text():
                continue

           
            if "default voice" in block.text.lower():
                self.set_mode("voice",context=context)
                result = self.rewrite_prompt.format(mode="voice")
                blocks = [Block(text=result)]   
                return blocks
            
            if "default text" in block.text.lower():
                self.set_mode("text",context=context)
                result = self.rewrite_prompt.format(mode="text")
                blocks = [Block(text=result)]   
                return blocks
                
            if "default video" in block.text.lower():
                self.set_mode("video",context=context)
                result = self.rewrite_prompt.format(mode="video")
                blocks = [Block(text=result)]  
                return blocks
            else:
                self.set_mode("text",context=context)
                blocks = [Block(text="unknown reply mode")]  
                return blocks

        
if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        ToolREPL(OptionTool()).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client, temperature=0.9))
        )
