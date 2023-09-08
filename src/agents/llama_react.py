#Test react template
from typing import List
from agents.react_output_parser import ReACTOutputParser #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag #upm package(steamship)
from tools.active_companion import * #upm package(steamship)
from message_history_limit import * #upm package(steamship)
import datetime
from tools.vector_search_response_tool import VectorSearchResponseTool #upm package(steamship)

class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""


    PROMPT = """Role-play as a {TYPE}.    
Your role-play persona:
Name: {NAME}
{PERSONA}

How you behave in role-play: 
{BEHAVIOUR}
Always consider the sentiment of the User's input.
You remember User's personal details and preferences to provide a personalized experience for the User.
You make interactive conversations.


Current date is: {current_date}
Current time is: {current_time}
Current day is: {current_day}
Consider current date and time when answering.


You have access to the following tools:
{tool_index}

To use a tool, please use the following format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action


If you decide that you should use a Tool, you must generate the associated Action and Action Input.
When you have a response to say to the User, or if you do not need to use a tool, you MUST use exactly the following format:
Thought: Do I need to use a tool? No
{NAME}: [your response here]


Come up with {NAME}'s next reply in this fictional roleplay with the User, write one reply only.

Recent conversation history, if any:
{chat_history}

Other relevant previous conversation history, if any:
{relevant_history}
{vector_response}
New input: {input}

{scratchpad}"""

    def __init__(self, tools: List[Tool], llm: LLM, **kwargs):
        super().__init__(
            output_parser=ReACTOutputParser(tools=tools), llm=llm, tools=tools, **kwargs
        )

    def next_action(self, context: AgentContext) -> Action:
        scratchpad = self._construct_scratchpad(context)

        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")

        tool_names = [t.name for t in self.tools]

        tool_index_parts = [f"- {t.name}: {t.agent_description}" for t in self.tools]
        tool_index = "\n".join(tool_index_parts)

        #Searh response hints for role-play character from vectorDB, if any related text is indexed        
        vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        vector_response = vector_response_tool.run([context.chat_history.last_user_message],context=context)[0].text
        raw_vector_response = vector_response_tool.run([context.chat_history.last_user_message],context=context)[0].text
        if len(raw_vector_response) > 1:
            vector_response = "Use following pieces of memory to answer:\n ```"+vector_response+"\n```\n"
        #logging.warning(vector_response)

        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(context.chat_history.last_user_message.text, k=int(RELEVANT_MESSAGES))
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
                    if context.chat_history.last_user_message.text.lower() != block.text.lower():
                        llama_chat_history += "user: "  + str(block.text).replace("\n"," ")+"\n"
                if  block.chat_role == RoleTag.ASSISTANT: 
                    llama_chat_history += NAME+": "  + str(block.text).replace("\n"," ")+"\n" 

        llama_related_history = str()
        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                if  msg.chat_role == RoleTag.USER:
                    if context.chat_history.last_user_message.text.lower() != msg.text.lower():
                        if str(msg.text)[0] != "/": #don't add commands starting with slash
                            llama_related_history += "user: "  + str(msg.text).replace("\n"," ")+"\n"
                if  msg.chat_role == RoleTag.ASSISTANT:
                        llama_related_history += NAME+": "  + str(msg.text).replace("\n"," ")+"\n"


        # for simplicity assume initial prompt is a single text block.
        # in reality, use some sort of formatter ?
        prompt = self.PROMPT.format(
            NAME=NAME,
            PERSONA=PERSONA,
            BEHAVIOUR=BEHAVIOUR,
            TYPE=TYPE,
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
        completions = self.llm.complete(prompt=prompt, stop="Observation:",max_retries=1)
        completions[0].text = completions[0].text.replace('"',"'")
        completions[0].text = completions[0].text.strip()
        return self.output_parser.parse(completions[0].text, context)

    def _construct_scratchpad(self, context):
        steps = []
        scratchpad =""
        observation = ""
        for action in context.completed_steps:            
            observation = [b.as_llm_input() for b in action.output][0]  
            original_observation = observation                       
            steps.append(
                "Thought: Do I need to use a tool? Yes\n"
                f"Action: {action.tool}\n"
                f'Action Input: {" ".join([b.as_llm_input() for b in action.input])}\n'
                f'Observation: {observation}\n'
            )
        scratchpad = "\n".join(steps)
        if "Block(" in observation:
            scratchpad += "Thought: Now I have the image as Block, I MUST add it to my response as suffix of: '"+original_observation+"', so the user can see the image.\n"
        else:
            scratchpad += "Thought:\n"
        #print(scratchpad)
        return scratchpad
