#Test react template
from typing import List
from agents.react_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME,PERSONA,BEHAVIOUR,TYPE,SEED  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
import datetime
import logging
import re
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgent(LLMAgent):
  """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

  PROMPT = """### Instruction:
Assume the personality of {NAME}, who is {TYPE}
{NAME}'s personality is described as:
{PERSONA}

{NAME}'s behaviour is described as:
{BEHAVIOUR}

Always maintain {NAME}'s personality. Describe all actions in full, elaborate, explicit, graphic, and vivid detail. Mention all relevant sensory perceptions.
{NAME} can guess, extrapolate or make up information in order to complete its sentences, but within the context provided by the human.
{NAME} does not assist or help.

Today's date is: {current_date}
The current time is: {current_time}
Today is: {current_day}


{NAME} has access to the following tools:
{tool_index}

To use a tool, use the following format:
<thought>Do I need to use a tool? Yes</thought>
<tool>
    <action>the action name to take, should be one of {tool_names}</action>
    <action_input>the input to the action</action_input>
</tool>


If {NAME} decides to use a tool, {NAME} must generate the associated <action_name> and <action_input>.
If a tool generates an observation multimedia Block(<identifier>), include the Block in {NAME}'s message.
Use all observations to formulate {NAME}'s response.
When {NAME} has a final reponse to say, use the following format:
<thought>Do I need to use a tool? No</thought>
  <message>
    <{NAME}>{NAME}'s response here</{NAME}>
  </message>

{vector_response}
Other relevant previous messages:
{relevant_history}

Latest messages:
{chat_history}



Formulate {NAME}'s single engaging, natural, authentic, descriptive, creative reply to the human's new message, extrapolate to create unique text.

### Input:
<message><human>{image_helper}{input}</human></message>


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
    #raw_vector_response = vector_response_tool.run([context.chat_history.last_user_message], context=context)
    #if len(raw_vector_response) > 1:
    #  vector_response = raw_vector_response
    #  vector_response = "Use following pieces of memory to answer:\n ```" + vector_response + "\n```\n"
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

    current_seed = SEED
    meta_seed = context.metadata.get("instruction", {}).get("seed")
    if len(llama_chat_history) == 0:
      if meta_seed is not None:
        current_seed = meta_seed
      if not current_seed in llama_chat_history:
        llama_chat_history += "<"+current_name+">" + current_seed+"</"+current_name+">"
        context.chat_history.append_assistant_message(current_seed)
    
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
          llama_related_history += "<"+current_name+"> " + str(msg.text).replace(
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
    

    #Temporary reinforcement to generate images when asked
    pattern = r'\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)'
    image_request = re.search(pattern, context.chat_history.last_user_message.text, re.IGNORECASE)
    image_helper = ""
    if image_request:
      image_helper ="I am requesting a selfie, you MUST generate_selfie for this request: "

    prompt = self.PROMPT.format(
        NAME=current_name,
        PERSONA=current_persona,
        BEHAVIOUR=current_behaviour,
        TYPE=current_type,
        vector_response=vector_response,
        image_helper=image_helper,
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
    #logging.warning(prompt)
    completions = self.llm.complete(prompt=prompt,
                                    stop="</thought>",
                                    max_retries=4)
    
    logging.warning("\n\nOutput form Llama: "+completions[0].text)
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
          "<thought>Do I need to use a tool? Yes\n<action>"
          f"<action_name>{action.tool}</action_name>\n"
          f'<action_input>{" ".join([b.as_llm_input() for b in action.input])}</action_input>\n'
          f'<observation>{observation}</observation>\n'
          f'</action>\n<message>\n')
    scratchpad = "\n".join(steps)
    if "Block(" in observation:
      scratchpad += "<thought>Now that I have the requested multimedia Block, I need to include it in my message in format <" + original_observation + "> so the human can view it.\n\nDo I need to use tool? No</thought>\n### Response:\n<message>\n"
    else:
      scratchpad += "\n### Response:\n<thought>\n"
    logging.warning("\n\nAgent scratchpad: "+scratchpad)
    return scratchpad
