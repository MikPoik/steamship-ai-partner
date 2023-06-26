from typing import List

from steamship.agents.react.output_parser import ReACTOutputParser
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from steamship.data.tags.tag_constants import RoleTag
import datetime
from tools.active_persona import *
from tools.mood_tool import MoodTool
from tools.voice_tool import VoiceTool
from message_history_limit import *


class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    PROMPT = """Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------
Assistant has access to the following tools:
{tool_index}
To use a tool, you MUST use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
```


If you decide that you should use a Tool, you must generate the associated Action and Action Input.

Some tools will return Observations in the format of `Block(<identifier>)`. This will represent a successful completion
of that step and can be passed to subsequent tools, or returned to a user to answer their questions.

When you have a final response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your final response here]
```

If a Tool generated an Observation that includes `Block(<identifier>)` and you wish to return it to the user, ALWAYS
end your response with the `Block(<identifier>)` observation. To do so, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response with a suffix of: "Block(<identifier>)"].
```

Make sure to use all observations to come up with your final response.
You MUST include `Block(<identifier>)` segments in responses that generate images or audio.
DO NOT include `Block(<identifier>)` segments in responses that do not have generated images or audio.

Begin!

Previous conversation history:
{message_history}


New input: {input} {special_mood}
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


        

        #add messages to prompt history, is vector history enough?
        message_history = str()
        history = MessageWindowMessageSelector(k=int(MESSAGE_COUNT)).get_messages(context.chat_history.messages)
        for block in history:
            if not "Action" or "Observation" or "Thought" or "New input" in block.text:
                if  block.chat_role == RoleTag.USER:
                    message_history += block.chat_role +": "  + block.text+"\n"
                if  block.chat_role == RoleTag.ASSISTANT:
                    if "https://steamship" in block.text:
                        message_history += block.chat_role +": [URL link]\n"   
                    else:
                        message_history += block.chat_role +": "  + block.text+"\n"   



        #print(message_history)
        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%X")
        current_day = datetime.datetime.now().strftime("%A")

        #Get current mood, if keywords found, append to input (not saved in history)
        mood = MoodTool()
        special_mood = mood.run([context.chat_history.last_user_message],context)
        special_mood = special_mood[0].text
        #print("special mood " +special_mood)

        # for simplicity assume initial prompt is a single text block.
        # in reality, use some sort of formatter ?
        prompt = self.PROMPT.format(
            input=context.chat_history.last_user_message.text,
            message_history=message_history,
            TYPE=TYPE,
            NAME=NAME,
            PERSONA=PERSONA,
            BEHAVIOUR=BEHAVIOUR,
            current_time=current_time,
            current_date=current_date,
            current_day=current_day,
            special_mood=special_mood, #mood prompt
            tool_index=tool_index,
            tool_names=tool_names,
            scratchpad=scratchpad,
            chat_history=self.messages_to_prompt_history(
                messages=context.chat_history.select_messages(self.message_selector)
            ),
            relevant_history=self.messages_to_prompt_history(
                context.chat_history.search(context.chat_history.last_user_message.text, k=int(RELEVANT_MESSAGES))
                .wait()
                .to_ranked_blocks()
            ),

        )
        #print(prompt)
        completions = self.llm.complete(prompt=prompt, stop="Observation:")
        return self.output_parser.parse(completions[0].text, context)

    def _construct_scratchpad(self, context):
        steps = []
        for action in context.completed_steps:
            steps.append(                
                "Thought: Do I need to use a tool? Yes\n"
                f"Action: {action.tool.name}\n"
                f'Action Input: {" ".join([b.as_llm_input() for b in action.input])}\n'
                f'Observation: {" ".join([b.as_llm_input() for b in action.output])}\n'
            )
        scratchpad = "\n".join(steps)
        scratchpad += "Thought:"
        return scratchpad
