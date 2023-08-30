"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
import sys,base64
import requests
from steamship import Block, Task, MimeTypes
from steamship.data.block import BlockUploadType
from steamship.agents.schema import AgentContext
from steamship.agents.tools.base_tools import ImageGeneratorTool
from steamship.agents.tools.image_generation.stable_diffusion import StableDiffusionTool
from steamship.utils.repl import ToolREPL
from steamship import File, Tag,DocTag
from tools.active_persona import *
#NSFW_SELFIE_TEMPLATE_PRE =""


#NSFW_SELFIE_TEMPLATE_POST = ""

class send_image(ImageGeneratorTool):

    name: str = "send_image"
    human_description: str = "Generates a selfie-style image from text with getimg.ai"
    agent_description = (
        "Used to generate a images from short text prompts, that describes how you look. Only use if the user has asked for a image "
        "When using this tool, the input should be a short plain text string of comma separated words that describes how you look."
    )

    generator_plugin_handle: str = "getimg-ai"
    generator_plugin_config: dict = {"api_key": "key-"}
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

    def run(
        self, tool_input: List[Block], context: AgentContext,api_key:str =""
    ) -> Union[List[Block], Task[Any]]:


        image_generator = context.client.use_plugin(
                    plugin_handle=self.generator_plugin_handle, config=self.generator_plugin_config
                )
        options={
            "width": 768,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
        }
        prompt = tool_input[0].text.replace(NAME,"") #don't add name
        prompt = prompt.replace("selfie","") #selfie doesn't work for current picture size
        task = image_generator.generate(
                    text=NSFW_SELFIE_TEMPLATE_PRE+tool_input[0].text.replace(NAME,"")+NSFW_SELFIE_TEMPLATE_POST,
                    make_output_public=True,
                    append_output_to_file=True,
                    options=options,
                )
        task.wait()
        blocks = task.output.blocks
        output_blocks = []
        for block in blocks:
            output_blocks.append(block)
        return output_blocks        
 
if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(send_image()).run()