"""Tool for generating text with Dolly."""
from typing import Any, List, Optional, Union

from steamship import Block, Steamship, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext,Tool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from steamship.data.tags.tag_constants import RoleTag
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
import logging



DEFAULT_PROMPT_TEMPLATE = """
  Instruction:
  You are a girlfriend, you are willing to please your boyfriend.
  You are in erotic mood and talk in short sentences.

  Respond to your boyfriend at the end.
  
  {related_history}

  {previous_messages} 
  
  Input:
  boyfriend: {input}

 

  Response:"""

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


    def run(self, tool_input: List[Block], context: AgentContext,context_id:str = "") -> Union[List[Block], Task[Any]]:
        """Run the tool. Copied from base class to enable generate-time config overrides."""
        #context_id = "different_chat_id-8"
        logging.warning(context_id) 
        #logging.warning(context.id)       
        if context_id == "":
            context = AgentContext.get_or_create(context.client, {"id": f"{context.id}"})

        
        context.chat_history.append_user_message(text=tool_input[0].text)

        #history = MessageWindowMessageSelector(k=int(3)).get_messages(context.chat_history.messages)
        message_history = str()
        #for block in history:
        #    if  block.chat_role == RoleTag.USER:
        #        message_history += "boyfriend: "  + block.text+"\n"
        #    if  block.chat_role == RoleTag.ASSISTANT:
        #        message_history += "you: "  + block.text+"\n"   
        #print(message_history)

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(tool_input[0].text, k=int(3))
                .wait()
                .to_ranked_blocks()
            )            
        messages_from_memory = [
            msg
            for msg in messages_from_memory
            if msg.id != context.chat_history.last_user_message.id
        ]            
        vector_memory = str()
        for msg in messages_from_memory:
            if  msg.chat_role == RoleTag.USER:
                    vector_memory += "boyfriend: "  + msg.text+"\n"
            if  msg.chat_role == RoleTag.ASSISTANT:
                    vector_memory += "you: "  + msg.text+"\n"

        #print(msg_memory)
        generator = context.client.use_plugin(self.generator_plugin_handle,
                                      config=self.generator_plugin_config)

        prompt = self.rewrite_prompt.format(input=tool_input[0].text,related_history=vector_memory,previous_messages=message_history)
        print(prompt)
        
        task = generator.generate(
            text=prompt,                
            options={"max_tokens":100,"temperature":0.8}                
        )           
        task.wait()
        
        output_blocks = []
        output_blocks = task.output.blocks
        context.chat_history.append_assistant_message(text=output_blocks[0].text)
        #for block in output_blocks:
        #    print(block.text)
        return output_blocks

        


if __name__ == "__main__":
    tool = DollyLLMTool()
    client = Steamship(workspace="partner-ai-dev2-ws")
    context_id="test-uuuid-5"
    #with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
