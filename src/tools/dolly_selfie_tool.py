"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
import logging,json
from tools.active_persona import NSFW_SELFIE_TEMPLATE


#Attempt to get Dolly output specific json
DOLLY_PROMPT_TEMPLATE = """
   ### Instruction: 
  You are the woman in the image, extract the image of you below . You have the image already. Using a keyword list describe the image and what are you wearing or if naked, don't output anything else.

   Input: {input}
    
    <image_keywords> describe image with keywords, separated by commas, use 3 to 5 keywords
   ### Response:
    <image_keywords>
   ### End"""

class DollySelfieTool(Tool):
    """Tool to generate images from text using"""

    rewrite_prompt = "Beautiful woman,{input},"+ NSFW_SELFIE_TEMPLATE
    dolly_rewrite_prompt = DOLLY_PROMPT_TEMPLATE

    name: str = "DollySelfieTool"
    human_description: str = "Generates images."
    agent_description = (
        "Used to generate images from text prompts. Only use if the user has asked directly for an image. "
        "When using this tool, the input should be a plain text string that describes, "
        "in detail, the desired image."
    )
    generator_plugin_handle: str = "replicate-kandinsky"
    generator_plugin_config: dict = {"replicate_api_key" : ""}
    dolly_generator_plugin_handle: str = "replicate-dolly-llm"


    def run(self, tool_input: List[Block], context: AgentContext,context_id:str = "",api_key="") -> Union[List[Block], Task[Any]]:
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

        print(dolly_output_blocks[0].text)
        
        kandinsky_input = dolly_output_blocks[0].text
        kandinsky_input = kandinsky_input.replace("<image_keywords>","")
        kandinsky_input = kandinsky_input.replace("</image_keywords>","")
        #return dolly_output_blocks
    
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)
        kandinsky_input = kandinsky_input.replace("\n","")
        prompt = self.rewrite_prompt.format(input=kandinsky_input)
        print(prompt)
        task = generator.generate(
            text=prompt,  
            make_output_public=True,
            append_output_to_file=True,              
            options={"num_inference_steps" : 50,
                     "num_steps_prior":15,
                     "height": 768,
                     "width":768
                     }                
        )           
        task.wait()
        
        output_blocks = []
        output_blocks.append(task.output.blocks[0])
        #print(str(output_blocks))        

        return output_blocks

        


if __name__ == "__main__":
    tool = DollySelfieTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
