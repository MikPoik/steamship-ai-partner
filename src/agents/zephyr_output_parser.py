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
        text = text.rstrip("'")  # remove trailing '
        text = text.lstrip("'")  #Remove leading '

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        if "(" + current_name + ":" in text:
            if not "action" in text.lower():
                return FinishAction(output=ReACTOutputParser._blocks_from_text(
                    context.client, text, context),
                                    context=context)

        #regex = r"action:\s*(.*?)\)\s§*action_input:\s*(.*?)\)"
        regex = '\(Action:\s*([^)/]*)\s*/?\)\s*\(Action_input:\s*([^)/]*)\s*/?\)'

        match = re.search(regex, text.lower(),
                          re.DOTALL | re.MULTILINE | re.IGNORECASE)

        if not match:
            logging.warning(f"Prefix missing, {text} send output to user..")
            text = text.replace(current_name + ":", "").strip()
            return FinishAction(output=ReACTOutputParser._blocks_from_text(
                context.client, text, context),
                                context=context)
        action = match.group(1)
        action = action.rstrip("'")
        action = action.lstrip("'")
        action = action.rstrip("//")
        action = action.rstrip("/")
        action_input = match.group(2).strip()
        action_input = action_input.rstrip("'")
        action_input = action_input.lstrip("'")
        action_input = action_input.rstrip("//")
        action_input = action_input.rstrip("/")
        tool = action.strip()
        tool = tool.lstrip("'")
        tool = tool.rstrip("'")
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
        regex = r"\({}:(.*?)\/\)".format(current_name)
        matches = re.findall(regex, message, re.DOTALL | re.MULTILINE)
        regex2 = r"\({}:(.*?)\)".format(current_name)
        matches2 = re.findall(regex2, message, re.DOTALL | re.MULTILINE)
        if matches:
            message = matches[0]
            message = message.strip()
            message = message.lstrip("'")
            message = message.rstrip("'")
        elif matches2:
            message = matches2[0]
            message = message.strip()
            message = message.lstrip("'")
            message = message.rstrip("'")
            #logging.warning(message)

        result_blocks: List[Block] = []

        block_found = 0
        block_id_regex = r"(?:(?:\[|\(|<)?Block)?\(?([A-F0-9]{8}\-[A-F0-9]{4}\-[A-F0-9]{4}\-[A-F0-9]{4}\-[A-F0-9]{12})\)?(?:(\]|\)|>)?)"
        remaining_text = message
        while remaining_text is not None and len(remaining_text) > 0:
            match = re.search(block_id_regex, remaining_text)
            if match:
                block_found = 1
                pre_block_text = ReACTOutputParser._remove_block_prefix(
                    candidate=remaining_text[0:match.start()])
                if len(pre_block_text) > 0:
                    result_blocks.append(Block(text=pre_block_text))
                result_blocks.append(Block.get(client, _id=match.group(1)))
                remaining_text = ReACTOutputParser._remove_block_suffix(
                    remaining_text[match.end():])
            else:
                if block_found == 0:
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
                else:
                    #final cleanup
                    remaining_text = remaining_text.lstrip("'")
                    remaining_text = remaining_text.rstrip("'")
                    result_blocks.append(Block(text=remaining_text))
                remaining_text = ""
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
