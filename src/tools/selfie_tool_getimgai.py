"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
from steamship.agents.schema import AgentContext #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from tools.active_companion import * #upm package(steamship)
from steamship import Block, Steamship, Task #upm package(steamship)
#NSFW_SELFIE_TEMPLATE_PRE =""


#NSFW_SELFIE_TEMPLATE_POST = ""

class SelfieTool(ImageGeneratorTool):

    name: str = "SelfieTool"
    human_description: str = "Generates a selfie-style image from text with getimg.ai"
    agent_description = (
        "Used to generate images from text prompts. Only use if the user has asked for an image, selfie or picture. "
        "When using this tool, the input should be a plain text string that describes, "
        "in detail, the desired image."
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
            "width": 512,
            "height": 768,
            "steps": 25,
            "guidance": 7.5,
        }
        prompt = tool_input[0].text.replace(NAME+",","")
        prompt = prompt.replace(NAME,"")
        prompt = prompt.replace('"',"")
        prompt = prompt.replace("'","")
        prompt = NSFW_SELFIE_TEMPLATE_PRE+prompt+NSFW_SELFIE_TEMPLATE_POST        
        task = image_generator.generate(
                    text=prompt,
                    make_output_public=True,
                    append_output_to_file=True,
                    options=options,
                )
        task.wait()
        blocks = task.output.blocks
        #output_blocks = [Block(text="You successfully sent the image. Do not describe the image, just respond to user.")]
        output_blocks = []
        #for func in context.emit_funcs:
        #    func(blocks, context.metadata)

        for block in blocks:
            output_blocks.append(block)
        return output_blocks        
 
if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(SelfieTool()).run()