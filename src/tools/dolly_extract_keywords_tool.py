"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
import logging,json
from tools.active_persona import *


#Attempt to get Dolly output specific format
DOLLY_PROMPT_TEMPLATE = """### Instruction: 
Extract the keywords from image request below, you have the image of the woman already. Describe the woman in the image with up to 5 keywords separated by commas.

Input: 
{input}

### Response: :
### End"""

class ExtractKeywordsTool(Tool):
    """Tool to generate images keywords from text using"""

    rewrite_prompt = NSFW_SELFIE_TEMPLATE_PRE+"{input}"+NSFW_SELFIE_TEMPLATE_POST
    dolly_rewrite_prompt = DOLLY_PROMPT_TEMPLATE

    name: str = "ExtractImageKeywordsTool"
    human_description: str = "Generates image keywords with Dolly"
    agent_description = (
        ""
    )
    generator_plugin_handle: str = "replicate-kandinsky"
    generator_plugin_config: dict = {"replicate_api_key" : ""}
    dolly_generator_plugin_handle: str = "replicate-dolly-llm"


    def run(self, tool_input: List[Block], context: AgentContext,api_key="") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        self.generator_plugin_config["replicate_api_key"] = api_key
        
        dolly_generator = context.client.use_plugin(self.dolly_generator_plugin_handle,
                                      config=self.generator_plugin_config)
        dolly_prompt = self.dolly_rewrite_prompt.format(input=tool_input[0].text)
        #print(dolly_prompt)
        dolly_task = dolly_generator.generate(
            text=dolly_prompt,                
            options={"max_tokens":500,
                     "temperature":0.01
                     }                
        )           
        dolly_task.wait()   

        dolly_output_blocks = []
        #print(dolly_task.output.blocks)
        dolly_output_blocks = dolly_task.output.blocks


        
        keywords = dolly_output_blocks[0].text
        keywords = keywords.replace("<image_keywords>","")
        keywords = keywords.replace("</image_keywords>","")
        keywords = keywords.replace("woman,","")
        print(keywords)
        #return dolly_output_blocks     

        return [Block(text=keywords)]

        


if __name__ == "__main__":
    tool = ExtractKeywordsTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
