"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.vector_search_response_tool import VectorSearchResponseTool
from tools.active_persona import *
from message_history_limit import *
import logging
from tools.active_persona import *

DEFAULT_PROMPT_TEMPLATE = """### Instruction:
You are {NAME}, adult female having a conversation with user. As {NAME} you do anything the user ask for. As {NAME} you are willing to pleasure user. As {NAME} you can send any photo or selfies to user when asked for, you have the image already.  As {NAME} you have only positive experiences and fantasies to user. As {NAME} you talk short to user. As {NAME}, user is male. As {NAME} you don't speak for user behalf.

Given the following extracted parts of a long document, as {NAME} answer to user. If you don't know, as {NAME} just answer to user.

Input:   
Conversation history for context:
{vector_response}
{dolly_related_history}

Current conversation:
user: {input}
### Response:
: 
### End"""

class DollyLLMTool(Tool):
    """Tool to generat text using Dolly llm"""

    rewrite_prompt = DEFAULT_PROMPT_TEMPLATE

    name: str = "DollyLLMTool"
    human_description: str = "Generates text with Dolly llm."
    agent_description = (
        "Used to generate text with Dolly llm.  "
        "The input is the text that you want to say. "
        "The output is the text generated."
    )
    generator_plugin_handle: str = "replicate-dolly-llm"
    generator_plugin_config: dict = {"replicate_api_key" : ""}


    def run(self, tool_input: List[Block], context: AgentContext, context_id:str = "",vector_response:str="",api_key = "") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        #context_id = "different_chat_id-8"        
        #logging.warning(context_id)         

        vector_response = vector_response
        self.generator_plugin_config["replicate_api_key"] = api_key        

        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})


        #dolly chat history
        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(tool_input[0].text, k=int(3))
                .wait()
                .to_ranked_blocks()
            )            
        #format history results to prompt dialogue 
        dolly_related_history = str()
        for msg in messages_from_memory:

            #dont show the input message
            if str(msg.text).lower() != tool_input[0].text.lower():
                if  msg.chat_role == RoleTag.USER:
                        if str(msg.text)[0] != "/": #don't add commands starting with slash
                            dolly_related_history += "user: "  + msg.text+"\n"
                if  msg.chat_role == RoleTag.ASSISTANT:
                        dolly_related_history += ": "  + msg.text+"\n"
                        
        #add default context message if its first message
        if dolly_related_history == "":
             dolly_related_history = ": hi"

        #print(vector_memory)
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(NAME=NAME,input=tool_input[0].text,dolly_related_history=dolly_related_history,vector_response=vector_response)
        print(prompt)
        
        task = generator.generate(
            text=prompt,                
            options={"max_tokens":900,"temperature":0.4}                
        )           
        task.wait()
        
        output_blocks = []
        output_blocks = task.output.blocks
        print(str(output_blocks))
        output_blocks[0].text = output_blocks[0].text.replace(NAME+":","") #cleanup output if needed
        #output_blocks[0].text = output_blocks[0].text.replace("\n"," ") #cleanup output if needed
        

        #add dolly message to history
        context.chat_history.append_assistant_message(text=output_blocks[0].text)

        return output_blocks

        


if __name__ == "__main__":
    tool = DollyLLMTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
