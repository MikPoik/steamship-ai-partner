"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
from steamship.agents.schema import AgentContext  #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship, Task  #upm package(steamship)
import logging
#NSFW_SELFIE_TEMPLATE_PRE =""

#NSFW_SELFIE_TEMPLATE_POST = ""


class SelfieTool(ImageGeneratorTool):

  name: str = "SelfieTool"
  human_description: str = "Generates a selfie-style image from text with getimg.ai"
  agent_description = (
      "This tool transforms text prompts into images. It should be employed only when the user requests an image, selfie, picture, etc. The input should be a detailed, plain text string describing the desired image.")

  generator_plugin_handle: str = "getimg-ai"
  generator_plugin_config: dict = {
      "api_key":
      "key-"
  }
  url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

  def run(self,
          tool_input: List[Block],
          context: AgentContext,
          api_key: str = "") -> Union[List[Block], Task[Any]]:

    image_generator = context.client.use_plugin(
        plugin_handle=self.generator_plugin_handle,
        config=self.generator_plugin_config)
    options = {
        "width": 384,
        "height": 512,
        "steps": 25,
        "guidance": 7.5,
    }
    
    current_name = NAME
    meta_name = context.metadata.get("instruction", {}).get("name")
    if meta_name is not None:
        current_name = meta_name

    prompt = tool_input[0].text.replace(NAME + ",", "")
    prompt = prompt.replace(current_name, "")
    prompt = prompt.replace('"', "")
    prompt = prompt.replace("'", "")

    pre_prompt = NSFW_SELFIE_TEMPLATE_PRE
    post_prompt = NSFW_SELFIE_TEMPLATE_POST

    meta_pre_prompt = context.metadata.get("instruction", {}).get("selfie_pre")
    if meta_pre_prompt is not None:
        pre_prompt = meta_pre_prompt

    meta_post_prompt = context.metadata.get("instruction", {}).get("selfie_post")
    if meta_post_prompt is not None:
        pre_prompt = meta_post_prompt    

    prompt = pre_prompt + prompt + post_prompt
    #logging.warning("Getimg prompt: "+prompt)
    task = image_generator.generate(
        text=prompt,
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
  ToolREPL(SelfieTool()).run()
