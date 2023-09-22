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
You are now embodying the personality of {NAME}, {TYPE}
{PERSONA}
{BEHAVIOUR}

Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}

You have access to the following tools:
{tool_index}

If you decide to use a Tool, generate the associated Action and Action Input. Use the following format, separated by triple backticks:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
```

If you have a final response for the Human, or if you do not need to use a tool, use the following format, separated by triple backticks:
```
Thought: Do I need to use a tool? No
{NAME}: [insert your final response here]
```

When crafting a unique reply from {NAME} to the new user message, consider the message history for topic. However, avoid directly repeating previous messages.

Here are some previous messages for context:
{relevant_history}

Here is the recent message history for context:
{chat_history}

{vector_response}
New message from user: {input}

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
            llama_chat_history += "user: " + str(block.text).replace(
                "\n", " ") + "\n"
        if block.chat_role == RoleTag.ASSISTANT:
          if block.text != "":
            llama_chat_history += current_name+": " + str(block.text).replace(
                "\n", " ") + "\n"

    meta_seed = context.metadata.get("instruction", {}).get("seed")
    if meta_seed is not None and llama_chat_history == "":
      llama_chat_history += current_name+": " + meta_seed
      context.chat_history.append_assistant_message(meta_seed)

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
              llama_related_history += "user: " + str(msg.text).replace(
                  "\n", " ") + "\n"
        if msg.chat_role == RoleTag.ASSISTANT:
          llama_related_history += current_name+": " + str(msg.text).replace(
              "\n", " ") + "\n"

     
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
                                    stop="Observation:",
                                    max_retries=4)
    #print(completions[0].text+"\n")
    return self.output_parser.parse(completions[0].text, context)

  def _construct_scratchpad(self, context):
    steps = []
    scratchpad = ""
    observation = ""
    for action in context.completed_steps:
      observation = [b.as_llm_input() for b in action.output][0]
      original_observation = observation
      if "Block(" in observation:
        observation = "The "+action.tool+" has generated an image represented by the "+original_observation
      steps.append(
          "Thought: Do I need to use a tool? Yes\n"
          f"Action: {action.tool}\n"
          f'Action Input: {" ".join([b.as_llm_input() for b in action.input])}\n'
          f'Observation: {observation}\n')
    scratchpad = "\n".join(steps)
    if "Block(" in observation:
      scratchpad += "Thought: Now that I have the image as Block, I MUST to include it in my final response as a suffix of [" + original_observation + "] so the image is visible.\n### Response:\n"
    else:
      scratchpad += "Thought:\n### Response:\n"
    print(scratchpad)
    return scratchpad
