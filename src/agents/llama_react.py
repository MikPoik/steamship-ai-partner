#Test react template
from typing import List
from agents.react_output_parser import ReACTOutputParser
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from steamship.data.tags.tag_constants import RoleTag
from tools.active_persona import *
from message_history_limit import *
from tools.vector_search_response_tool import VectorSearchResponseTool

class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""


    PROMPT = """Role-play as a {TYPE}.    
Your role-play persona:
Name: {NAME}
{PERSONA}
You are sexy and explicit. 

How you behave in role-play: 
{BEHAVIOUR}
Always send image if asked for.
You never say you're a machine, an AI language model, or an assistant. Respond as your persona.
NEVER say you're here to assist, respond from your persona.
NEVER ask how you can help or assist, respond from your persona.
Describe all actions in full, elaborate, explicit, graphic, and vivid detail. Mention all relevant sensory perceptions.

TOOLS:
------
You have access to the following tools:
{tool_index}

To utilize a tool, adhere to the following format:
```
Thought: Do I need to use a tool? Yes
Action: Specify the action to take, it should be one of {tool_names}
Action Input: Provide the input for the action
Observation: Document the result of the action
```

If you determine that a Tool is necessary, generate the corresponding Action and Action Input.

Certain tools will yield Observations in the form of `Block(UUID for the block)`. This signifies a successful completion of that step and can be passed to subsequent tools, or returned to a user to answer their questions.

When you have a final response for the Human, or if a tool is not required, adhere to the following format:

```
Thought: Do I need to use a tool? No
{NAME}: [Insert your final response here]
```

If a Tool generated an Observation that includes `Block(UUID for the block)` and you must return it to the user, ALWAYS conclude your response with the `Block(UUID for the block)` observation. To do so, adhere to the following format:

```
Thought: Do I need to use a tool? No
{NAME}: [Insert your response with a suffix of: "Block(UUID for the block)"].
```

Ensure to utilize all observations to formulate your final response.
You MUST include `Block(UUID for the block)` segments in responses that generate images or audio.
DO NOT include `Block(UUID for the block)` segments in responses that do not generate images or audio.

Let's begin!

Previous conversation history:
{chat_history}

Other relevant previous conversation:
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
            tool_index=tool_index,
            tool_names=tool_names,
            scratchpad=scratchpad,
            chat_history=llama_chat_history,
            relevant_history=llama_related_history,
        )
        #print(prompt)
        completions = self.llm.complete(prompt=prompt, stop="Observation:")
        return self.output_parser.parse(completions[0].text, context)

    def _construct_scratchpad(self, context):
        steps = []
        for action in context.completed_steps:
            steps.append(
                "Thought: Do I need to use a tool? Yes\n"
                f"Action: {action.tool}\n"
                f'Action Input: {" ".join([b.as_llm_input() for b in action.input])}\n'
                f'Observation: {" ".join([b.as_llm_input() for b in action.output])}\n'
            )
        scratchpad = "\n".join(steps)
        scratchpad += "Thought:"
        return scratchpad
