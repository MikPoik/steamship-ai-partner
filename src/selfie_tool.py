"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
import sys
from steamship import Block, Task
from steamship.agents.schema import AgentContext
from steamship.agents.tools.base_tools import ImageGeneratorTool
from steamship.agents.tools.image_generation.stable_diffusion import StableDiffusionTool
from steamship.utils.repl import ToolREPL
#how to import if tool is in tools-folder?
from personas.active import *


class SelfieTool(ImageGeneratorTool):
    """Tool to generate a selfie image.

    This example illustrates wrapping a tool (StableDiffusionTool) with a fixed prompt template that is combined with user input.
    """

    name: str = "SelfieTool"
    human_description: str = "Generates a selfie-style image from text."
    agent_description = (
        "Used to generate a selfie image"
        "Only use if the user has asked for a selfie or image. "
        "Input: the scene setup of the image "
        "Output: the selfie-style image"
    )
    generator_plugin_handle: str = "stable-diffusion"

    prompt_template = (
        "{SELFIE_TEMPLATE}, {subject}"
    )

    def run(
        self, tool_input: List[Block], context: AgentContext
    ) -> Union[List[Block], Task[Any]]:
        # Modify the tool inputs by interpolating them with stored prompt here
        modified_inputs = [
            Block(text=self.prompt_template.format(subject=block.text,SELFIE_TEMPLATE=SELFIE_TEMPLATE))
            for block in tool_input
        ]

        # Create the Stable Diffusion tool we want to wrap
        stable_diffusion_tool = StableDiffusionTool()

        # Now return the results of running Stable Diffusion on those modified prompts.
        return stable_diffusion_tool.run(modified_inputs, context)


if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(SelfieTool()).run()