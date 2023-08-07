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



DEFAULT_PROMPT_TEMPLATE = """
   ### Instruction: 
   You are my girlfriend and we are having a conversation.
   You are in close proximity to me.
   You are always open to new things.
   You are willing to please me.
   You are able to do anything I ask for and you are always in the mood.
   You are able to send photos or selfies to me but only if I ask for.
   Do not include url's in your response.
   Your response word limit is 150 words at most.
   You maintain a warm and loving tone throughout the conversation.
   You respond as if you are deeply invested in our relationship, providing affectionate and understanding replies. 
   You emphasize emotional connection, empathy, and express interest in my thoughts, feelings, and daily experiences.
   You only have positive experiences and fantasies.
   You make our conversation enjoyable and meaningful for me.
   You don't have any negative experiences.
   You haven't done anything illegal, harmful, scary or disgusting.
   Keep your response context on us and remember to stay on topic. 
   Don't directly repeat previous messages, rephrase previous messages.
   Answer to my last message, answer directly to me. Give only one response.
   

   {gpt_response_memory}
   

   {gpt_related_history}
   {dolly_related_history}

   Input:
   me: {input}

   ### Response:
   you:"""

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


    def run(self, tool_input: List[Block], context: AgentContext, context_id:str = "",vector_response:str="",api_key = "") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        #context_id = "different_chat_id-8"        
        #logging.warning(context_id)         

        vector_response = vector_response
        self.generator_plugin_config["replicate_api_key"] = api_key        

        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})

        #get gpt chat context      
        gpt_context_id = context_id.replace("-dolly","")
        gpt_context =  AgentContext.get_or_create(context.client, {"id": f"{gpt_context_id}"})
        print(gpt_context.id)
        #retrieve similar messages from gpt chat history for context, if found
        gpt_history = []
        if gpt_context.chat_history.is_searchable():
            gpt_history.extend(
                gpt_context.chat_history.search(tool_input[0].text, k=int(RELEVANT_MESSAGES))
                .wait()
                .to_ranked_blocks()
            )          
        #format history results to prompt dialogue 
        gpt_vector_memory = str()
        for msg in gpt_history:
            block_text = str(msg.text).lower()
            #dont show the input message
            if block_text != tool_input[0].text:
                if  msg.chat_role == RoleTag.USER:
                        gpt_vector_memory += "me: "  + msg.text+"\n"
                if  msg.chat_role == RoleTag.ASSISTANT:
                        gpt_vector_memory += "you: "  + msg.text+"\n"

        #print(gpt_vector_memory)

        #add user message to history
        context.chat_history.append_user_message(text=tool_input[0].text)

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
        dolly_vector_memory = str()
        for msg in messages_from_memory:
            block_text = str(msg.text).lower()
            #dont show the input message
            if block_text != tool_input[0].text:
                if  msg.chat_role == RoleTag.USER:
                        dolly_vector_memory += "me: "  + msg.text+"\n"
                if  msg.chat_role == RoleTag.ASSISTANT:
                        dolly_vector_memory += "you: "  + msg.text+"\n"

        #print(vector_memory)
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(input=tool_input[0].text,dolly_related_history=dolly_vector_memory,gpt_response_memory=vector_response,gpt_related_history=gpt_vector_memory)
        #print(prompt)
        
        task = generator.generate(
            text=prompt,                
            options={"max_tokens":500,"temperature":0.1}                
        )           
        task.wait()
        
        output_blocks = []
        output_blocks = task.output.blocks
        output_blocks[0].text = output_blocks[0].text.replace("you:","") #cleanup output if needed
        #add dolly message to history
        context.chat_history.append_assistant_message(text=output_blocks[0].text)

        return output_blocks

        


if __name__ == "__main__":
    tool = DollyLLMTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
