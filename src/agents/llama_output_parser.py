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
        ReACTOutputParser.tools_lookup_dict = {
            tool.name: tool
            for tool in kwargs.pop("tools", [])
        }
        super().__init__(**kwargs)

    def parse(self, response: Dict, context: AgentContext) -> Action:
        text = {}
        run_tool = {}
        run_tool_input = {}
        if response is not None:
            text = response.get("response", {})
            run_tool = response.get("run_tool", {})
            run_tool_input = response.get("run_tool_input", {})

        text = text.strip()  #remove extra spaces

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name


        return FinishAction(output=ReACTOutputParser._blocks_from_text(
            context.client, text, run_tool, run_tool_input,
            context),
                            context=context)

    @staticmethod
    def _blocks_from_text(client: Steamship, text: str, tool_name: str,
                          tool_input: str,
                          context: AgentContext) -> List[Block]:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        result_blocks: List[Block] = []

        block_found = 0

        if len(text) > 0:
            result_blocks.append(Block(text=text))

        saved_block = context.metadata.get("blocks", {}).get("image")
        if saved_block is not None:
            #print("get block from metadata")
            result_blocks.append(Block.get(client, _id=saved_block))
            context.metadata['blocks'] = None
        else:

            create_images = context.metadata.get("instruction",
                                                 {}).get("create_images")
            if create_images == "true":
                if tool_name:
                    selfie_tool = ReACTOutputParser.tools_lookup_dict.get(
                        tool_name, None)
                    if selfie_tool:
                        # Call the tool with the input
                        image_block = selfie_tool.run([Block(
                            text=tool_input)], context) if tool_input else None
                        if image_block:
                            result_blocks.append(image_block[0])
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
