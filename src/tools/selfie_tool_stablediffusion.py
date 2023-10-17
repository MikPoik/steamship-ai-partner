"""STABLE DIFFUSION Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
import sys
from steamship import Block, Task  #upm package(steamship)
from steamship.agents.schema import AgentContext  #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool  #upm package(steamship)
from steamship.agents.tools.image_generation.stable_diffusion import StableDiffusionTool  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from tools.active_companion import SELFIE_TEMPLATE  #upm package(steamship)

NEGATIVE_PROMPT = "bad anatomy, bad composition, ugly, abnormal, unrealistic, double, contorted, disfigured, malformed, amateur, extra, duplicate,2 heads,2 faces"


class SelfieToolSD(ImageGeneratorTool):
  """Tool to generate a selfie image.

    This example illustrates wrapping a tool (StableDiffusionTool) with a fixed prompt template that is combined with user input.
    """

  name: str = "SelfieTool"
  human_description: str = "Generates a selfie-style image from text."
  agent_description = (
      "Used to generate a image from text, that describes the scene setting of the image."
      "Only use if the user has asked for a image "
      "Input: Imagine the photo scene of the image where it is taken, use comma separated list of keywords"
      "Output: the generated image")
  generator_plugin_handle: str = "stable-diffusion"
  generator_plugin_config: dict = {"n": 1, "inference_steps": 25}

  prompt_template = ("{SELFIE_TEMPLATE}, {subject}")

  def run(self, tool_input: List[Block],
          context: AgentContext) -> Union[List[Block], Task[Any]]:
    # Modify the tool inputs by interpolating them with stored prompt here
    modified_inputs = [
        Block(text=self.prompt_template.format(
            subject=block.text, SELFIE_TEMPLATE=SELFIE_TEMPLATE))
        for block in tool_input
    ]

    image_generator = context.client.use_plugin(
        plugin_handle="stable-diffusion", config={
            "n": 1,
            "size": "768x768"
        })

    task = image_generator.generate(
        text=modified_inputs[0].text,
        make_output_public=True,
        append_output_to_file=True,
        options={
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 7,
            "num_inference_steps": 25,
        },
    )
    task.wait()
    blocks = task.output.blocks
    output_blocks = []
    for block in blocks:
      output_blocks.append(block)
    return output_blocks


if __name__ == "__main__":
  print("Try running with an input like 'penguin'")
  ToolREPL(SelfieToolSD()).run()
