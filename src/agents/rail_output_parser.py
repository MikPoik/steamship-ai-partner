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

    def parse(self, response: str, function_call: Dict,
              context: AgentContext) -> Action:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name
        text = ""
        run_tool = {}
        run_tool_input = {}
        if response is not None:
            text = response.split("{")[0] #split possible json
            text = text.split("#")[0] #split possible hashtags
            text = text.replace(current_name + "`", "")
            if "(function_call" in text:
                text = text.split("(function_call")[0]
            text = re.sub(re.escape("xoxo"), "", text, flags=re.IGNORECASE)
            text = re.sub(re.escape(current_name) + r": ", "", text, flags=re.IGNORECASE)
            
            text = text.strip()
            # Remove a single quote sign from text, if it is present
            if text.count('"') == 1:
                text = text.replace('"', '')
            if text.startswith("'"):
                text = text[1:]                
            if text.endswith('['):  # Check if text ends with "["
                text = text[:-1]  # Remove it
            if text.endswith('('):  # Check if text ends with "("
                text = text[:-1]  # Remove it
            if text.count('"') == 2:
                if text.startswith('"') and text.endswith('"'):
                    text = text.lstrip('"')
                    text = text.rstrip('"')
                


                
            if function_call is not None:
                run_tool_func = function_call.get("function_call")
                if run_tool_func is not None:
                    run_tool = run_tool_func.get("name", "")
                    #print("run_tool", run_tool)
                    run_tool_input = run_tool_func.get("tool_input", "")
                    #print("run_tool_input", run_tool_input)
                    
                    run_tool_text = function_call.get("response")
                    if run_tool_text is not None:
                        text = run_tool_text
                        text = text.split("[")[0] #split possible extra
                        text = text.split("#")[0] #split possible extra
                        text = text.split("(")[0] #split possible extra
                        text = re.sub(re.escape("xoxo"), "", text, flags=re.IGNORECASE)
                        text = text.replace(current_name + ": ", "")
                        text = text.strip()
        

        return FinishAction(output=ReACTOutputParser._blocks_from_text(
            context.client, text, run_tool, run_tool_input, context),
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
            image_request = False
            if create_images == "true":
                if tool_name:
                    selfie_tool = ReACTOutputParser.tools_lookup_dict.get(
                        tool_name, None)
                    if selfie_tool:
                        image_request = True
                        #print(tool_input)
                        if tool_input:
                            # Call the tool with the input
                            str_tool_input = ','.join(tool_input)
                            image_block = selfie_tool.run(
                                [Block(text=str_tool_input)], context
                            ) if tool_input else None  # Pass image description as string
                            if image_block:
                                result_blocks.append(image_block[0])
                if not image_request:                    
                    #Another way to check for image generation, if agent forgets to use a tool
                    pattern = r'.*\b(?:here|sent|takes)\b(?=.*?(?:selfie|picture|photo|image|peek)).*'
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    if compiled_pattern.search(text):
                      check_image_block = context.metadata.get("blocks",
                                                               {}).get("image")
                      if check_image_block is None:
                        #logging.warning("Create selfie for prompt")
                        create_images = context.metadata.get("instruction",
                                                             {}).get("create_images")
                        if create_images == "true":
                          selfie = SelfieTool()
                          image_block = selfie.run([Block(text=text)],
                                                   context)
                          result_blocks.append(image_block[0])
        return result_blocks
