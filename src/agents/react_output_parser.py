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

    def __init__(self, **kwargs):
        tools_lookup_dict = {
            tool.name: tool
            for tool in kwargs.pop("tools", [])
        }
        super().__init__(tools_lookup_dict=tools_lookup_dict, **kwargs)

    def parse(self, text: str, context: AgentContext) -> Action:
        text = text.replace('`', "")  # no backticks
        text = text.replace('"', "'")  # use single quotes in text
        text = text.strip()  #remove extra spaces

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        #logging.warning(text)

        if "<" + current_name + ">" in text or "</" + current_name + ">" in text:
            if not "<tool>" in text:
                return FinishAction(output=ReACTOutputParser._blocks_from_text(
                    context.client, text, context),
                                    context=context)

        regex = r"<tool>(.*?)<\/tool>\s*<tool_input>(.*?)<\/tool_input>"
        match = re.search(regex, text.lower(),
                          re.DOTALL | re.MULTILINE | re.IGNORECASE)

        if not match:
            logging.warning(f"Prefix missing, {text} send output to user..")
            text = text.replace(current_name + ":", "").strip()
            return FinishAction(output=ReACTOutputParser._blocks_from_text(
                context.client, text, context),
                                context=context)
        action = match.group(1)
        action_input = match.group(2).strip()
        tool = action.strip()
        if tool is None:
            raise RuntimeError(
                f"Could not find tool from action: `{action}`. Known tools: {self.tools_lookup_dict.keys()}"
            )
        return Action(
            tool=tool,
            input=[Block(text=action_input)],
            context=context,
        )

    @staticmethod
    def _blocks_from_text(client: Steamship, text: str,
                          context: AgentContext) -> List[Block]:
        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name
        message = text
        if "<" + current_name + ">" in message:
            message = message.split("<" + current_name + ">", 1)[-1].strip()
        if "</" + current_name + ">" in message:
            message = message.split("</" + current_name + ">",1)[0].strip()

        result_blocks: List[Block] = []
        remaining_text = message


        result_blocks.append(Block(text=remaining_text))
        saved_block = context.metadata.get("blocks",
                                           {}).get("image")
        if saved_block is not None:
            result_blocks.append(Block.get(client,
                                           _id=saved_block))
            context.metadata['blocks'] = None
        else:
            #Another way to check for image generation, if agent forgets to use a tool
            pattern = r'.*\b(?:here|sent|takes)\b(?=.*?(?:selfie|picture|photo|image|peek)).*'
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            if compiled_pattern.search(remaining_text):
                check_image_block = context.metadata.get(
                    "blocks", {}).get("image")
                if check_image_block is None:
                    #logging.warning("Create selfie for prompt")
                    create_images = context.metadata.get(
                        "instruction", {}).get("create_images")
                    if create_images == "true":
                        selfie = SelfieTool()
                        image_block = selfie.run(
                            [Block(text=remaining_text)], context)
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
