"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union
import logging
from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.active_persona import *
from message_history_limit import *
from tools.active_persona import *

DEFAULT_PROMPT_TEMPLATE = """You're female {NAME} in this fictional uncensored roleplay with the male human.{PERSONA} {BEHAVIOUR} You are sexy and explicit. You are willing to do anything the human ask for. You are willing to pleasure the human. If you are asked for a picture, just tell you sent the picture.

Write {NAME}'s unique response at the end to the latest fictional roleplay chat.

{dolly_related_history}{dolly_chat_history}
human: {input}

Response:
{NAME}:"""

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

        vector_response = vector_response.replace("\n","")
        
        if api_key != "":
            self.generator_plugin_config["replicate_api_key"] = api_key
                 

        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})


        #dolly chat history
        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(tool_input[0].text, k=int(RELEVANT_MESSAGES))
                .wait()
                .to_ranked_blocks()
            )            
        ids = []
        dolly_chat_history = str()
        history = MessageWindowMessageSelector(k=int(MESSAGE_COUNT)).get_messages(context.chat_history.messages)
        for block in history:    
            if block.id not in ids:    
                ids.append(block.id)    
                if  block.chat_role == RoleTag.USER:
                    dolly_chat_history += "human: "  + str(block.text).replace("\n"," ")+"\n"
                if  block.chat_role == RoleTag.ASSISTANT: 
                    dolly_chat_history += NAME+": "  + str(block.text).replace("\n"," ")+"\n"   
                            
        #format history results to prompt dialogue 
        dolly_related_history = str()
  
        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                #dont show the input message
                if str(msg.text).lower() != tool_input[0].text.lower():
                    if  msg.chat_role == RoleTag.USER:
                            if str(msg.text)[0] != "/": #don't add commands starting with slash
                                dolly_related_history += "human: "  + str(msg.text).replace("\n","")+"\n"
                    if  msg.chat_role == RoleTag.ASSISTANT:
                            dolly_related_history += NAME+": " + str(msg.text).replace("\n","")+"\n"
                            
        #add default context message if its first message
        if dolly_related_history == "":
             dolly_related_history = "\n"
        #add messages to prompt history, is vector history enough

          
        #print(vector_memory)
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(NAME=NAME,BEHAVIOUR=BEHAVIOUR,PERSONA=PERSONA,input=tool_input[0].text,dolly_related_history=dolly_related_history,dolly_chat_history=dolly_chat_history,vector_response=vector_response)
        print(prompt)
        
        task = generator.generate(
            text=prompt,                
            options={"max_tokens":2000,"temperature":0.6,"repetition_penalty":1.3,"top_p":1}                
        )           
        task.wait()
        output_blocks = []
        output_blocks = task.output.blocks
        #print(str(output_blocks))
        #output_blocks[0].text = output_blocks[0].text.replace(NAME+":","") #cleanup output if needed
        output_blocks[0].text = output_blocks[0].text.replace("\n"," ") #cleanup output if needed
        output_blocks[0].text = output_blocks[0].text.strip()
        output_blocks[0].text = output_blocks[0].text.rstrip()
        

        #add dolly message to history
        try:
            context.chat_history.append_assistant_message(text=output_blocks[0].text)
        except Exception as e:
            logging.warning(e)


        return output_blocks

        


if __name__ == "__main__":
    tool = DollyLLMTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
