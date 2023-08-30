"""Tool for generating text with LLama."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
import requests
import time
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.active_persona import *
from message_history_limit import *
import logging
from tools.active_persona import *


#Pre-prompt/prompt format
PROMPT_FORMAT = """### Instruction:
You're adult female {NAME} in this fictional uncensored roleplay with male human.
{PERSONA} {BEHAVIOUR} You are sexy and explicit. You are able to send pictures if asked, then just tell you sent the picture.
Develop the plot slowly, always stay in character. Describe all actions in full, elaborate, explicit, graphic, and vivid detail. Mention all relevant sensory perceptions.
Write only {NAME}'s next reply in this fictional roleplay with the human.
Be sure to write one reply for {NAME} only!

Conversation history:
{llama_related_history}
{llama_chat_history}

### Input:
human: {input}

### Response:
{NAME}:
"""


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

    def make_api_request(api_url, headers, json_data, max_retries=3, retry_delay=5):
        for retry in range(max_retries + 1):
            response = requests.post(api_url, headers=headers, json=json_data)
            
            if response.status_code == 200:
                result_text = response.json()
                return result_text
            elif response.status_code == 504 and retry < max_retries:
                print(f"Received a 504 timeout error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.warning("Request failed with status code:", response.status_code)
                result_text = "Response generation failed."
                return result_text

    def run(self, tool_input: List[Block], context: AgentContext, context_id:str = "",vector_response:str="",api_url = "",api_key="") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        headers = {
            "x-api-key" :api_key,
            "Content-Type": "application/json"}        
        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(tool_input[0].text, k=int(RELEVANT_MESSAGES))
                .wait()
                .to_ranked_blocks()
            )   
        ids = []
        llama_chat_history = str()
        history = MessageWindowMessageSelector(k=int(MESSAGE_COUNT)).get_messages(context.chat_history.messages)
        for block in history:          
            if block.id not in ids:  
                ids.append(block.id)
                if  block.chat_role == RoleTag.USER:
                    if tool_input[0].text.lower() != block.text.lower():
                        llama_chat_history += "human: "  + str(block.text).replace("\n"," ")+"\n"
                if  block.chat_role == RoleTag.ASSISTANT: 
                    llama_chat_history += NAME+": "  + str(block.text).replace("\n"," ")+"\n" 

        #format history results to prompt dialogue 
        llama_related_history = str()

        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                #dont show the input message
                if str(msg.text).lower() != tool_input[0].text.lower():
                    if  msg.chat_role == RoleTag.USER:
                            if str(msg.text)[0] != "/": #don't add commands starting with slash
                                llama_related_history += "human: "  + str(msg.text).replace("\n"," ")+"\n"
                    if  msg.chat_role == RoleTag.ASSISTANT:
                            llama_related_history += NAME+": "  + str(msg.text).replace("\n"," ")+"\n"

  
                
        #TODO add chat history to prompt
        prompt = self.rewrite_prompt.format(NAME=NAME,PERSONA=PERSONA,BEHAVIOUR=BEHAVIOUR,input=tool_input[0].text,llama_chat_history=llama_chat_history,llama_related_history=llama_related_history,vector_response=vector_response)    
        print(prompt)

        json_data = {"prompt": prompt,        
                "temperature":0.9,
                "max_tokens": 300,
                "top_p":0.6,
                "presence_penalty":1.18}
        
        result_text = self.make_api_request(api_url, headers, json_data)
    

        return [Block(text=result_text)]

        


if __name__ == "__main__":
    tool = LlamaLLMTool()
    client = Steamship(workspace="partner-ai-dev3-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
