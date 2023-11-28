import logging
import re
from typing import Dict, List, Optional
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, FinishAction, OutputParser, Tool  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
import re


class ReACTOutputParser(OutputParser):
    'Parse LLM output expecting structure matching ReACTAgent default prompt'

    tools_lookup_dict: Optional[Dict[str, Tool]] = None
    gen_image = False

    def __init__(self, **kwargs):
        tools_lookup_dict = {
            tool.name: tool
            for tool in kwargs.pop("tools", [])
        }
        super().__init__(tools_lookup_dict=tools_lookup_dict, **kwargs)

    def parse(self, text: str, context: AgentContext) -> Action:
        #text = text.replace('`', "")  # no backticks
        #text = text.replace('"', "'")  # use single quotes in text
        text = text.replace('</s>', "")  # remove
        text = text.replace('<|user|>', "")  # remove
        text = text.replace('Human:', "")

        text = text.strip()  #remove extra spaces
        text = text.rstrip("'")  # remove trailing '
        text = text.lstrip("'")  #Remove leading '

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        text = text.replace(f'{current_name}:', "")  # remove
        return FinishAction(output=ReACTOutputParser._blocks_from_text(
            context.client, text, context),
                            context=context)

    @staticmethod
    def _blocks_from_text(client: Steamship, text: str,
                          context: AgentContext) -> List[Block]:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        # New regex to match description within angle brackets starting with "Image:"
        image_match = re.search(r"<Image:\s*(.*?)>", text)
        image_description = ""
        remaining_text = text
        if image_match:
            image_description = image_match.group(1).strip()
            # Remove the image description text from the remaining text
            remaining_text = text.replace(image_match.group(0), '').strip()

        result_blocks: List[Block] = []
        #print("image_description ", image_description)
        block_found = 0

        if len(remaining_text) > 0:
            result_blocks.append(Block(text=remaining_text))

        saved_block = context.metadata.get("blocks", {}).get("image")
        if saved_block is not None:
            #print("get block from metadata")
            result_blocks.append(Block.get(client, _id=saved_block))
            context.metadata['blocks'] = None
        else:
            #print("check for image")
            #Another way to check for image generation, if agent forgets to use a tool
            pattern = r'.*\b(?:here|sent|send|sends|takes)\b(?=.*?(?:selfie|picture|photo|image|peek)).*'
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            if compiled_pattern.search(
                    remaining_text) or len(image_description) > 0:
                #print("generate image")
                check_image_block = context.metadata.get("blocks",
                                                         {}).get("image")
                if check_image_block is None:
                    #logging.warning("Create selfie for prompt")
                    create_images = context.metadata.get(
                        "instruction", {}).get("create_images")
                    if create_images == "true":
                        selfie = SelfieTool()
                        tool_input = ""
                        if len(image_description) > 0:
                            tool_input = image_description
                        else:
                            tool_input = remaining_text
                        image_block = selfie.run([Block(text=tool_input)],
                                                 context)
                        result_blocks.append(image_block[0])

                #final cleanup
                #result_blocks.append(Block(text=remaining_text))

        return result_blocks

    @staticmethod
    def _remove_block_prefix(candidate: str) -> str:
        removed = candidate
        if removed.endswith("(Block") or removed.endswith(
                "[Block") or removed.endswith("<Block"):
            removed = removed[len("Block") + 1:]
        elif removed.endswith("Block"):
            removed = removed[len("Block"):]
        return removed

    @staticmethod
    def _remove_block_suffix(candidate: str) -> str:
        removed = candidate
        if removed.startswith(")") or removed.endswith(
                "]") or removed.endswith(">"):
            removed = removed[1:]
        return removed
