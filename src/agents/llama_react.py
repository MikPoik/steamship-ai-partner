#Test react template
from typing import List
from agents.react_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME,PERSONA,BEHAVIOUR,TYPE  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
import datetime
import logging
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgent(LLMAgent):
  """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

  PROMPT = """### Instruction:
You are now embodying the human personality of {NAME}, who is {TYPE}
{NAME}'s personality can be described as:
{PERSONA}
{NAME}'s behaviour can be described as:
{BEHAVIOUR}

Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}

TOOLS:
------
You have access to the following tools:
{tool_index}

To use a tool, use the following format:

<thought>Do I need to use a tool? Yes</thought>
<tool>
<action>the action to take, should be one of {tool_names}</action>
<action_input>the input to the action</action_input>
</tool>

If you decide that you should use a Tool, you must generate the associated <action> and <action_input>.
If a tool generated a Block(<identifier>) multimedia observation, include the Block(<identifier) in response.
When you have a unique final response to the new user message, or you do not need to use a tool, respond with the following format:

<thought>Do I need to use a tool? No</thought>
<response>
<{NAME}>insert final response here in plain text</{NAME}>
</response>

### Input:
Here are some previous messages for context:
{relevant_history}

Here is the recent message history for context:
{chat_history}

{vector_response}
New message from user:
<human>{input}</human>


{scratchpad}"""

  def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
    super().__init__(output_parser=ReACTOutputParser(tools=tools),
                     llm=llm,
                     tools=tools,
                     **kwargs)

  def next_action(self, context: AgentContext) -> Action:
    scratchpad = self._construct_scratchpad(context)

    current_date = datetime.datetime.now().strftime("%x")
    current_time = datetime.datetime.now().strftime("%X")
    current_day = datetime.datetime.now().strftime("%A")

    current_name = NAME      
    meta_name = context.metadata.get("instruction", {}).get("name")
    if meta_name is not None:
      current_name = meta_name 


    tool_names = [t.name for t in self.tools]

    tool_index_parts = [
        f"- {t.name}: {t.agent_description}" for t in self.tools
    ]
    tool_index = "\n".join(tool_index_parts)

    #Searh response hints for role-play character from vectorDB, if any related text is indexed
    vector_response = ""
    raw_vector_response = ""
    vector_response_tool = VectorSearchResponseTool()
    raw_vector_response = vector_response_tool.run([context.chat_history.last_user_message], context=context)
    if len(raw_vector_response) > 1:
      vector_response = raw_vector_response
      vector_response = "Use following pieces of memory to answer:\n ```" + vector_response + "\n```\n"
    #logging.warning(vector_response)

    messages_from_memory = []
    # get prior conversations
    if context.chat_history.is_searchable():
      messages_from_memory.extend(
          context.chat_history.search(
              context.chat_history.last_user_message.text,
              k=int(RELEVANT_MESSAGES)).wait().to_ranked_blocks())
    ids = []
    llama_chat_history = str()
    history = context.chat_history.select_messages(self.message_selector)

    for block in history:
      if block.id not in ids:
        ids.append(block.id)
        if block.chat_role == RoleTag.USER:
          if context.chat_history.last_user_message.text.lower(
          ) != block.text.lower():
            llama_chat_history += "<human>" + str(block.text).replace(
                "\n", " ") + "</human>\n"
        if block.chat_role == RoleTag.ASSISTANT:
          if block.text != "":
            llama_chat_history += "<"+current_name+">" + str(block.text).replace(
                "\n", " ") + "</"+current_name+">\n"

    meta_seed = context.metadata.get("instruction", {}).get("seed")
    if meta_seed is not None and len(llama_chat_history) < 1:
      llama_chat_history += "<"+current_name+">" + meta_seed+"</"+current_name+">"
      #context.chat_history.append_assistant_message(meta_seed)

    llama_related_history = str()
    for msg in messages_from_memory:
      #don't add duplicate messages
      if msg.id not in ids:
        ids.append(msg.id)
        if msg.chat_role == RoleTag.USER:
          if context.chat_history.last_user_message.text.lower(
          ) != msg.text.lower():
            if str(
                msg.text)[0] != "/":  #don't add commands starting with slash
              llama_related_history += "<human>: " + str(msg.text).replace(
                  "\n", " ") + "</human>\n"
        if msg.chat_role == RoleTag.ASSISTANT:
          llama_related_history += current_name+": " + str(msg.text).replace(
              "\n", " ") + "</"+current_name+">\n"

     
    current_persona = PERSONA
    current_behaviour = BEHAVIOUR
    current_type = TYPE  

    meta_name = context.metadata.get("instruction", {}).get("name")
    if meta_name is not None:
      current_name = meta_name 

    meta_persona = context.metadata.get("instruction", {}).get("personality")
    if meta_persona is not None:
      current_persona = meta_persona

    meta_behaviour =  context.metadata.get("instruction", {}).get("behaviour")
    if meta_behaviour is not None:
      current_behaviour = meta_behaviour

    meta_type =  context.metadata.get("instruction", {}).get("type")
    if meta_type is not None:
      current_type = meta_type
    
    #logging.warning("Dynamic personality :"+current_name+"\n"+current_persona +"\n"+current_type+"\n"+current_behaviour)

    prompt = self.PROMPT.format(
        NAME=current_name,
        PERSONA=current_persona,
        BEHAVIOUR=current_behaviour,
        TYPE=current_type,
        vector_response=vector_response,
        input=context.chat_history.last_user_message.text,
        current_date=current_date,
        current_time=current_time,
        current_day=current_day,
        tool_index=tool_index,
        tool_names=tool_names,
        scratchpad=scratchpad,
        chat_history=llama_chat_history,
        relevant_history=llama_related_history,
    )
    #print(prompt)
    completions = self.llm.complete(prompt=prompt,
                                    stop="<observation>",
                                    max_retries=4)
    
    #logging.warning(completions[0].text)
    return self.output_parser.parse(completions[0].text, context)

  def _construct_scratchpad(self, context):
    meta_name = context.metadata.get("instruction", {}).get("name")
    if meta_name is not None:
      current_name = meta_name 
    else:
      current_name = NAME    
    steps = []
    scratchpad = ""
    observation = ""
    for action in context.completed_steps:
      observation = [b.as_llm_input() for b in action.output][0]
      original_observation = observation
      if "Block(" in observation:
        observation = original_observation
      steps.append(
          "<tool>\n"
          f"<action>{action.tool}</action>\n"
          f'<action_input>{" ".join([b.as_llm_input() for b in action.input])}</action_input>\n'
          f'<observation>{observation}</observation>\n'
          f'</tool>\n')
    scratchpad = "\n".join(steps)
    #if "Block(" in observation:
    #  scratchpad += "<thought>Now that I have the requested multimedia, I need to include it between {NAME}'s response tags in square brackets " + original_observation + "</thought>\n### Response:\n"
    #else:
    scratchpad += "\n<thought>\n"
    #print(scratchpad)
    return scratchpad
