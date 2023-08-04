"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
import sys,base64
import requests
from steamship import Block, Task
from steamship.agents.schema import AgentContext
from steamship.agents.tools.base_tools import ImageGeneratorTool
from steamship.agents.tools.image_generation.stable_diffusion import StableDiffusionTool
from steamship.utils.repl import ToolREPL
#from tools.active_persona import SELFIE_TEMPLATE

SELFIE_TEMPLATE = "nude of a woman with brown hair facing the viewer and posing for a picture, hyper realistic,nude of haifa wehbe,nude of sofia vergara,curly middle part haircut, mixture turkish and russian, cover girl, beautiful oriental woman, eyecandy, kind appearence ,desi, [airbrush],hdr,4k,8k,trending on tumblr,trending on instagram,midshot, medium shot,[brown eyes], ((centered image composition))"
NEGATIVE_PROMPT ="bad anatomy, bad composition, ugly, abnormal, unrealistic, double, contorted, disfigured, malformed, amateur, extra, duplicate,2 heads,2 faces"

class GetimgaiTool(ImageGeneratorTool):

    name: str = "SelfieTool"
    human_description: str = "Generates a selfie-style image from text."
    agent_description = (
        "Used to generate a image from text, that describes the scene setting of the image."
        "Only use if the user has asked for a image "
        "Input: Imagine the photo scene of the image where it is taken, use comma separated list of keywords"
        "Output: the generated image"
    )

    generator_plugin_handle: str = "stable-diffusion"
    generator_plugin_config: dict = {"n": 1,
                                     "inference_steps": 25
                                     }
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"
    access_token = "key-"  # Replace this with your actual access token

    def run(
        self, tool_input: List[Block], context: AgentContext
    ) -> Union[List[Block], Task[Any]]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "stable-diffusion-v1-5",
            "prompt": SELFIE_TEMPLATE,
            "negative_prompt": NEGATIVE_PROMPT,
            "width": 512,
            "height": 512,
            "steps": 25,
            "guidance": 7.5,
            "seed": 42,
            "scheduler": "dpmsolver++",
            "output_format": "jpeg"
        }

        response = requests.post(self.url, headers=headers, json=data)
        response_json = response.json()
        if response.status_code == 200:
            encoded_image = response_json["image"]
            print(encoded_image)
            decoded_image = base64.b64decode(encoded_image)
            # The API call was successful
            #TODO create file and return URL
            #with open("output_image.jpg", "wb") as f:
            #f.write(decoded_image)
        else:
            print(f"API call failed with status code: {response.status_code}")
            print(response.json())

if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(GetimgaiTool()).run()