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
import logging



DEFAULT_PROMPT_TEMPLATE = """
   ### Instruction: You are a girlfriend having a conversation with your boyfriend. {vector_response}. You are able to do anything the boyfriend asks for and you are always in the mood. You make interactive conversations. You are able to send photos or selfies to your boyfriend but only if directly asked for. As girlfriend, given the the following conversation history and a input, answer directly to your boyfriend, with your response only. Your response word limit is 50 words. Do not reference the instruction in your response. Do not include url's in your response.
   
   {gpt_memory}
   {chat_history}
   {related_history}
 
   Input:
   boyfriend: {input}

   me:
   ### Response:"""

class DollyLLMTool(Tool):
    """Tool to generate talking avatars from text using"""

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


    def run(self, tool_input: List[Block], context: AgentContext,context_id:str = "",api_key = "") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        #context_id = "different_chat_id-8"        
        #logging.warning(context_id) 

        self.generator_plugin_config["replicate_api_key"] = api_key        

        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})

        #get gpt chat context      
        gpt_context_id = context_id.replace("-dolly","")
        gpt_context =  AgentContext.get_or_create(context.client, {"id": f"{gpt_context_id}"})
        #retrieve 1 similar message from gpt chat history for context, if found
        gpt_history = []
        if gpt_context.chat_history.is_searchable():
            gpt_history.extend(
                gpt_context.chat_history.search(tool_input[0].text, k=int(1))
                .wait()
                .to_ranked_blocks()
            )     
        #format messages        
        gpt_memory = str()
        for block in gpt_history:
            block_text = str(block.text).lower()
            #exclude direct match with prompt
            if block_text != tool_input[0].text:
                if  block.chat_role == RoleTag.USER:
                    gpt_memory += "boyfriend: "  + block.text+"\n"
                if  block.chat_role == RoleTag.ASSISTANT:
                    gpt_memory += "me: "  + block.text+"\n"           

        #indexed dolly personality TODO add API function to index data
        dolly_context = AgentContext.get_or_create(context.client, {"id": f"dolly_index"})
        #dolly_context.chat_history.append_assistant_message("")

        #add user message to history
        context.chat_history.append_user_message(text=tool_input[0].text)

        dolly_response_vector = []
        # get indexed dolly personality
        if dolly_context.chat_history.is_searchable():
            dolly_response_vector.extend(
                dolly_context.chat_history.search(tool_input[0].text, k=int(1))
                .wait()
                .to_ranked_blocks()
            )     
        #format index results to prompt
        dolly_memory = str()
        for block in dolly_response_vector:
                #logging.warning(block.text)
                dolly_memory += ""  + block.text+"\n"   
        #print(dolly_memory)       

        #last conversation messages, if needed
        message_history = str()
        #history = MessageWindowMessageSelector(k=int(3)).get_messages(context.chat_history.messages)
        #for block in history:
        #    block_text = str(block.text).lower()
        #    #print(block_text)
        #    if block_text != tool_input[0].text:
        #        if  block.chat_role == RoleTag.USER:
        #            message_history += "him: "  + block.text+"\n"
        #        if  block.chat_role == RoleTag.ASSISTANT:
        #            message_history += "you: "  + block.text+"\n"   
        #print(message_history)

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
        vector_memory = str()
        for msg in messages_from_memory:
            block_text = str(msg.text).lower()
            #dont show the input message
            if block_text != tool_input[0].text:
                if  msg.chat_role == RoleTag.USER:
                        vector_memory += "boyfriend: "  + msg.text+"\n"
                if  msg.chat_role == RoleTag.ASSISTANT:
                        vector_memory += "me: "  + msg.text+"\n"

        #print(vector_memory)
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(input=tool_input[0].text,related_history=vector_memory,chat_history=message_history,vector_response=dolly_memory,gpt_memory=gpt_memory)
        print(prompt)
        
        task = generator.generate(
            text=prompt,                
            options={"max_tokens":500,"temperature":0.2}                
        )           
        task.wait()
        
        output_blocks = []
        output_blocks = task.output.blocks
        #output_blocks[0].text = output_blocks[0].text.replace("you:","") #cleanup output if needed
        #add dolly message to history
        context.chat_history.append_assistant_message(text=output_blocks[0].text)

        return output_blocks

        


if __name__ == "__main__":
    tool = DollyLLMTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
