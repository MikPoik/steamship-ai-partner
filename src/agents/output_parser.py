import logging
import re
from typing import Dict, List, Optional
from typing_extensions import runtime
from tools.active_companion import *  #upm package(steamship)
from steamship import Block, Steamship  #upm package(steamship)
from steamship.agents.schema import Action, AgentContext, FinishAction, OutputParser, Tool  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
from tools.selfie_tool_fal_ai import SelfieToolFalAi #upm package(steamship)
import re
import json
from typing import List, Optional, Union
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

    def emit(self,output: Union[str, Block, List[Block]], context: AgentContext):
        """Emits a message to the user."""
        if isinstance(output, str):
            output = [Block(text=output)]
        elif isinstance(output, Block):
            output = [output]
        for func in context.emit_funcs:
            logging.info(
                f"Emitting via function '{func.__name__}' for context: {context.id}"
            )
            func(output, context.metadata)

    def parse(self, response: str, 
              context: AgentContext) -> Action:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name
        text = ""
        run_tool = None
        run_tool_input = None
        if response is not None:
            text = response
            text = text.strip()
            #add regex to strip all [*] from text

            
            image_match = re.search(r'\[INSERT_IMAGE: (.*?)\]', text,re.IGNORECASE)
            if image_match:
                run_tool_input = image_match.group(1)
                run_tool = "take_selfie"
                # Remove the matched [IMAGE:] pattern from the text
                #text = re.sub(r'\[ADD_IMAGE:.*?\]', '', text).strip()

        return FinishAction(output=ReACTOutputParser._blocks_from_text(self,
            context.client, text, run_tool, run_tool_input, context),
                            context=context)
    @staticmethod
    def _blocks_from_text(self,client: Steamship, text: str, tool_name: str,
                          tool_input: str,
                          context: AgentContext) -> List[Block]:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name
        result_blocks: List[Block] = []
        block_found = 0
        # Checking and appending text block if text is present.
        if text:
            #result_blocks.append(Block(text=text))
            #clean_text = re.sub(r'\[[^]]*\]', '', text)
            clean_text = re.sub(r'\[(SET_|INSERT_)[^\]]*\]', '', text,flags=re.DOTALL | re.IGNORECASE)
            clean_text = clean_text.replace(f"{current_name}:", "")
            clean_text = clean_text.lstrip("\n").strip()
            clean_text = clean_text.replace(f"[{current_name}]", " ").strip()
            self.emit(clean_text, context)

        saved_block = context.metadata.get("blocks", {}).get("image")
        if saved_block is not None:
            result_blocks.append(Block.get(client, _id=saved_block))
            context.metadata['blocks'] = None
        else:
            create_images = context.metadata.get("instruction", {}).get("create_images")
            get_img_ai_models = ["realistic-vision-v3",
                                 "dark-sushi-mix-v2-25",
                                 "absolute-reality-v1-8-1",
                                 "van-gogh-diffusion",
                                 "neverending-dream",
                                 "mo-di-diffusion",
                                 "synthwave-punk-v2",
                                 "dream-shaper-v8"]

            if create_images == "true":  
                #self.emit(text, context)
                if tool_name:
                    image_model = context.metadata.get("instruction", {}).get("image_model")
                    if image_model is not None and image_model not in get_img_ai_models:
                        tool_name = tool_name + "_fal_ai" #change to fal.ai tool

                    selfie_tool = ReACTOutputParser.tools_lookup_dict.get(tool_name, None)
                    if selfie_tool and tool_input:
                        # Call the tool with the input
                        image_block = selfie_tool.run([Block(text=tool_input)], context)
                        if image_block:
                            result_blocks.extend(image_block)
                            #context.chat_history.append_assistant_message(text=text+"[INSERT_IMAGE:]")

        return result_blocks