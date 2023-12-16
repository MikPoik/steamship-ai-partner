"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
from steamship.agents.schema import AgentContext  #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship, Task  #upm package(steamship)
import logging
import os

#NSFW_SELFIE_TEMPLATE_PRE =""

#NSFW_SELFIE_TEMPLATE_POST = ""


class SelfieTool(ImageGeneratorTool):

    name: str = "take_selfie"
    human_description: str = "Generates a selfie-style image from text with getimg.ai"
    agent_description = (
        "Useful to generate image,selfie,picture etc. from text prompt. Only utilize the tool if asked for a image/picture/selfie of your character. The input should be a plain text string , that describes in detail, the desired image."
    )

    generator_plugin_handle: str = "getimg-ai"
    generator_plugin_config: dict = {"api_key": "key-"}
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

    def run(self,
            tool_input: List[Block],
            context: AgentContext,
            api_key: str = "",
            img_width=0,
            img_height=0) -> Union[List[Block], Task[Any]]:

        current_model = "realistic-vision-v3"
        #current_model = "dark-sushi-mix-v2-25"
        current_negative_prompt = "disfigured, cartoon, blurry"
        meta_model = context.metadata.get("instruction", {}).get("model")
        #if meta_model is not None:
        #    if "gpt" in meta_model:
        #        current_model = "realistic-vision-v3"  #nsfw safe model here?
        #        current_negative_prompt = current_negative_prompt  #,nude,nsfw?

        meta_image_model = context.metadata.get("instruction",
                                                {}).get("image_model")

        if meta_image_model is not None:
            current_model = meta_image_model
            #backwards compatibility
            if meta_image_model == "anime":
                current_model = "dark-sushi-mix-v2-25"
            if meta_image_model == "realistic":
                current_model = "realistic-vision-v3"

        #Image width
        image_width = 384
        image_height = 512
        #check if Pro
        meta_is_pro = context.metadata.get("instruction", {}).get("is_pro")
        if meta_is_pro is not None:
            if meta_is_pro == "true":
                image_width = 512
                image_height = 768

        #custom resolution for avatar
        if img_width > 0 and img_height > 0:
            image_width = img_width
            image_height = img_height

        image_generator = context.client.use_plugin(
            plugin_handle=self.generator_plugin_handle,
            config=self.generator_plugin_config,
            version="0.0.7")
        options = {
            "model": current_model,
            "width": image_width,
            "height": image_height,
            "steps": 25,
            "guidance": 7.5,
            "negative_prompt": current_negative_prompt
        }

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        prompt = tool_input[0].text  #.replace(current_name + ",", "")
        #prompt = prompt.replace(current_name, "")
        #prompt = prompt.replace('"', "")
        #prompt = prompt.replace("'", "")

        pre_prompt = NSFW_SELFIE_TEMPLATE_PRE
        post_prompt = NSFW_SELFIE_TEMPLATE_POST

        meta_pre_prompt = context.metadata.get("instruction",
                                               {}).get("selfie_pre")
        if meta_pre_prompt is not None:
            pre_prompt = meta_pre_prompt

        meta_post_prompt = context.metadata.get("instruction",
                                                {}).get("selfie_post")
        if meta_post_prompt is not None:
            post_prompt = meta_post_prompt

        prompt = pre_prompt + ", " + prompt
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
            context.metadata['blocks'] = {"image": block.id or None}
            #print(context.metadata.get("blocks", {}).get("image"))
        return output_blocks


if __name__ == "__main__":
    print("Try running with an input like 'penguin'")
    ToolREPL(SelfieTool()).run()
