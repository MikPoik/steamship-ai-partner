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


NEGATIVE_PROMPT ="disfigured, cartoon, blurry,two person"

class SelfieNSFWTool(ImageGeneratorTool):

    name: str = "SelfieTool"
    human_description: str = "Generates a selfie-style image from text with getimg.ai"
    agent_description = (
        "Used to generate a image from text, that describes the scene setting of the image."
        "Only use if the user has asked for a image "
        "Input: Imagine the photo scene of the image where it is taken, use comma separated list of keywords"
        "Output: the generated image"
    )

    generator_plugin_handle: str = ""
    generator_plugin_config: dict = {"n": 1,
                                     "inference_steps": 25
                                     }
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"
    access_token = ""  # Replace this with your actual access token

    def run(
        self, tool_input: List[Block], context: AgentContext,api_key:str =""
    ) -> Union[List[Block], Task[Any]]:
        if api_key != "":
            self.access_token = api_key


        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "realistic-vision-v3",
            "prompt": NSFW_SELFIE_TEMPLATE_PRE + tool_input[0].text + NSFW_SELFIE_TEMPLATE_POST,
            "negative_prompt": NEGATIVE_PROMPT,
            "width": 768,
            "height": 1024,
            "steps": 25,
            "guidance": 7.5,
            "scheduler": "dpmsolver++",
            "output_format": "png"
        }

        response = requests.post(self.url, headers=headers, json=data)
        response_json = response.json()
        if response.status_code == 200:
            encoded_image = response_json["image"]
            decoded_image = base64.b64decode(encoded_image)

            png_file = File.create(context.client,content=decoded_image,mime_type=MimeTypes.PNG)                    
            block = Block(content_url=png_file.raw_data_url,mime_type=MimeTypes.PNG,url=png_file.raw_data_url)    
            png_file.set_public_data(True)
            return [block]

        else:
            print(f"API call failed with status code: {response.status_code}")
            print(response.json())

if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(SelfieNSFWTool()).run()