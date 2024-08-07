"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
from steamship.agents.schema import AgentContext  #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship, Task  #upm package(steamship)
import logging
import os
import re

#NSFW_SELFIE_TEMPLATE_PRE =""

#NSFW_SELFIE_TEMPLATE_POST = ""


class SelfieTool(ImageGeneratorTool):

    name: str = "selfie_tool"
    human_description: str = "Useful to generate a image. The input should be a plain text string , that describes in detail, the desired image."
    agent_description = (
        "This tool generates a image, picture or selfie. The input should be a JSON object with list of uncensored keywords that describes the desired image. The output is an image that correlates with the input keywords. "
    )

    generator_plugin_handle: str = "getimg-ai-image-generator"
    generator_plugin_config: dict = {
        "api_key":
        "key-"
    }
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

    def strip_duplicate_words(self, prompt_words: List[str],
                              pre_prompt_words: List[str]) -> List[str]:
        """Strip duplicate words in prompt_words that already exist in pre_prompt_words."""
        stripped_prompt_words = [
            word for word in prompt_words if word not in pre_prompt_words
        ]
        return stripped_prompt_words

    def run(self,
            tool_input: List[Block],
            context: AgentContext,
            api_key: str = "",
            img_width=0,
            img_height=0,
            stream=True) -> Union[List[Block], Task[Any]]:

        current_model = "realistic-vision-v3"
        #current_model = "dark-sushi-mix-v2-25"
        current_negative_prompt = "disfigured,deformed, poorly drawn, extra limbs, blurry:0.25,((double image)), ((split image)), ((two images)),((two bodies)),text"
        negative_post = ""
        meta_current_level = context.metadata.get("instruction",
                                                  {}).get("level")
        if meta_current_level is not None:
            if int(meta_current_level) < 30:
                negative_post = ",((nude)),((naked)),((nsfw)),((uncensored))"

        meta_model = context.metadata.get("instruction", {}).get("model")

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
        image_width = 512
        image_height = 768
        #check if Pro
        meta_is_pro = context.metadata.get("instruction", {}).get("is_pro")
        if meta_is_pro is not None:
            if meta_is_pro == "true":
                image_width = 768
                image_height = 1024

        #custom resolution for avatar
        if img_width > 0 and img_height > 0:
            image_width = img_width
            image_height = img_height

        image_generator = context.client.use_plugin(
            plugin_handle=self.generator_plugin_handle,
            config=self.generator_plugin_config,
            version="1.0.2")
        options = {
            "model": current_model,
            "width": image_width,
            "height": image_height,
            "steps": 30,
            "guidance": 7,
            "negative_prompt": current_negative_prompt
        }

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        prompt = tool_input[0].text.replace('"', '')
        prompt = prompt.replace("Keywords:", "").strip()

        #print(prompt)

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

        current_type = TYPE
        meta_type = context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            if len(current_type) < 30:
                current_type = meta_type
            else:
                current_type = meta_type.split(".")[0]
        prompt_words = prompt.split(",")  # Split the prompt into words
        pre_prompt_words = pre_prompt.split(
            ",")  # Split the pre_prompt into words
        stripped_prompt_words = self.strip_duplicate_words(
            prompt_words, pre_prompt_words)

        prompt_with_parentheses = ', '.join(
            [f'(({word.strip()}))' for word in stripped_prompt_words])

        current_type_words = current_type.split(
            ",")  # Split the prompt into words
        current_type_with_brackets = ', '.join(
            [f'[{word.strip()}]' for word in current_type_words])

        pre_prompt_words = pre_prompt.split(",")  # Split the prompt into words
        pre_prompt_with_brackets = ', '.join(
            [f'[{word.strip()}]' for word in pre_prompt_words])

        meta_level = context.metadata.get("instruction", {}).get("level")
        if meta_level is not None and meta_level < 30:
            current_negative_prompt += negative_post
            pre_prompt_with_brackets = "((((with clothes))))," + pre_prompt_with_brackets
            options["negative_prompt"] = current_negative_prompt

        prompt = f"{prompt_with_parentheses},{pre_prompt_with_brackets}"
        if context.metadata.get("verbose_logging", False):
            logging.warning("** Running Getimg tool with prompt**\n" + prompt +
                            "\n" + current_negative_prompt + "**")
        task = image_generator.generate(
            text=prompt,
            make_output_public=True,
            append_output_to_file=True,
            output_file_id=context.chat_history.file.id,
            streaming=stream,
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
