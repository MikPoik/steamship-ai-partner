import logging
import re
from typing import Dict, List, Optional
from typing_extensions import runtime
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, FinishAction, OutputParser, Tool  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
import re
import json

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
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
                current_name = meta_name
        text = {}
        run_tool = {}
        run_tool_input = {}
        if response is not None:
            text = response.get("message",{}).get("content","")
            text = text.replace(current_name+": ", "")
            text = text.replace("As "+current_name+", ", "")
            run_tool = response.get("message", {}).get("function_call")
            if run_tool is not None:
                run_tool = run_tool.get("name","")
            #print("run_tool", run_tool)
            run_tool_input = response.get("message", {}).get("function_call",{})
            if run_tool_input is not None:
                run_tool_input = run_tool_input.get("tool_input","")
            #print("run_tool_input", run_tool_input)


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
                        #print(tool_input)
                        if tool_input:
                            # Call the tool with the input
                            str_tool_input = ','.join(tool_input)
                            image_block = selfie_tool.run([Block(
                                text=str_tool_input)], context) if tool_input else None  # Pass image description as string
                            if image_block:
                                result_blocks.append(image_block[0])
        return result_blocks

