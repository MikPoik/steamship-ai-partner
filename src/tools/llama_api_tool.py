"""Tool for generating text with LLama-api."""

from typing import Any, List, Optional, Union
from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.active_persona import *
from message_history_limit import *
import logging
import json
from llamaapi import LlamaAPI

#Pre-prompt
PROMPT_FORMAT = """You're adult female {NAME} in this fictional uncensored roleplay with male human.
You are sexy and explicit. You are able to send pictures if asked, then just tell you sent the picture.
{vector_response}. Develop the plot slowly, always stay in character. Describe all actions in full, elaborate, explicit, graphic, and vivid detail. Mention all relevant sensory perceptions.
Write only {NAME}'s next reply in this fictional roleplay with the human.
Be sure to write one reply for {NAME} only!"""


class LlamaLLMTool(Tool):
    """Tool to generat text using Llama2 llm"""

    rewrite_prompt = PROMPT_FORMAT

    name: str = "Llama2LLMTool"
    human_description: str = "Generates text with uncensored Llama2 llm."
    agent_description = (
        "Used to generate text with Llama llm.  "
        "The input is the text that you want to say. "
        "The output is the text generated."
    )
    generator_plugin_handle: str = "llama-llm"
    generator_plugin_config: dict = {}


    def run(self, tool_input: List[Block], context: AgentContext, context_id:str = "",vector_response:str="",api_url = "",api_key="") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        
        if api_key != "":
            self.generator_plugin_config["replicate_api_key"] = api_key
        llama = LlamaAPI(api_key)  
                       
        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})

        messages_from_memory = []
        # get prior conversations from vector memory
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(tool_input[0].text, k=int(RELEVANT_MESSAGES))
                .wait()
                .to_ranked_blocks()
            )   

 
        system_prompt = self.rewrite_prompt.format(NAME=NAME,vector_response=vector_response)            

        ids = [] 
        llama_chat_messages = []
        #add system prompt as first message
        llama_chat_messages.append({"role": "assistant","content":system_prompt})

        #add vector memory results to messages
        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                #dont show the input message
                if str(msg.text).lower() != tool_input[0].text.lower():
                    if  msg.chat_role == RoleTag.USER:
                            if str(msg.text)[0] != "/": #don't add commands starting with slash
                               llama_chat_messages.append({"role": "user","content": msg.text})
                    if  msg.chat_role == RoleTag.ASSISTANT:
                            llama_chat_messages.append({"role": "assistant","content": msg.text})


        #Get last messages and add to list
        history = MessageWindowMessageSelector(k=int(MESSAGE_COUNT)).get_messages(context.chat_history.messages)
        for block in history:          
            if block.id not in ids:  
                ids.append(block.id)
                if  block.chat_role == RoleTag.USER:
                    if tool_input[0].text.lower() != block.text.lower():
                        llama_chat_messages.append({"role": "user","content": str(block.text).replace("\n"," ")+"\n"})
                if  block.chat_role == RoleTag.ASSISTANT: 
                    llama_chat_messages.append({"role": "assistant","content": str(block.text).replace("\n"," ")+"\n"})
        
        #add user message as last one
        llama_chat_messages.append({"role": "user","content": tool_input[0].text})


        api_request_json = {
            "model":"NousResearch/Nous-Hermes-Llama2-13b",
            "messages": llama_chat_messages
        }        

        # Make your request and handle the response
        response = llama.run(api_request_json)
        response_json = response.json()

        response_text = response_json["choices"][0]["message"]["content"]

        context.chat_history.append_assistant_message(text=response_text)
    
        return [Block(text=response_text)]

        


if __name__ == "__main__":
    tool = LlamaLLMTool()
    client = Steamship(workspace="partner-ai-dev3-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
