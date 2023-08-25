"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from tools.active_persona import *
import logging



class SelfieTool(Tool):
    """Tool to generate images from text using"""

    rewrite_prompt = SELFIE_TEMPLATE_PRE+"{subject}"+SELFIE_TEMPLATE_POST 

    name: str = "SelfieTool"
    human_description: str = "Generates images from text with Kandinsky."
    agent_description = (
        "Used to generate images from text prompts. Only use if the user has asked directly for an image. "
        "When using this tool, the input should be a plain text string that describes,"
        "in detail, the desired image."
    )
    generator_plugin_handle: str = "replicate-kandinsky"
    generator_plugin_config: dict = {"replicate_api_key" : "your-api-key"}


    def run(self, tool_input: List[Block], context: AgentContext,context_id:str = "",api_key="") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        if api_key != "":
            self.generator_plugin_config["replicate_api_key"] = api_key
            
        modified_inputs = [
            Block(text=self.rewrite_prompt.format(subject=block.text))
            for block in tool_input
        ]
        #print(str(modified_inputs))
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(subject=modified_inputs)
        #print(prompt)
        task = generator.generate(
            text=prompt,
            make_output_public=True,
            append_output_to_file=True,                
            options={"num_inference_steps" : 75,
                     "num_steps_prior":25,
                     "height":1024,
                     "width": 768
                     }                
        )           
        task.wait()
        blocks = task.output.blocks
        output_blocks = []
        for block in blocks:
            output_blocks.append(block)
        return output_blocks

        


if __name__ == "__main__":
    tool = SelfieTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
