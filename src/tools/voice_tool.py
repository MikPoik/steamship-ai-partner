"""Tool for generating images."""
from steamship import Steamship
from steamship.agents.llms import OpenAI
from steamship.agents.tools.speech_generation import GenerateSpeechTool
from steamship.agents.utils import with_llm
from steamship.utils.repl import ToolREPL
from typing import Any, List, Union
from steamship import Block, Task
from steamship.agents.schema import AgentContext
from tools.active_persona import VOICE_ID




class VoiceTool(GenerateSpeechTool):
    """Tool to generate audio from text."""

    name: str = "GenerateSpokenAudio"
    human_description: str = "Generates spoken audio from text."
    agent_description: str = (
        "Use this tool ALWAYS to generate spoken audio from text, the input should be a plain text string containing the "
        "content to be spoken. "        
    )

    prompt_template = (
        "{subject}"
    )
    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:

        for block in tool_input:
            context.chat_history.append_assistant_message(block.text)
            
        modified_inputs = [
            Block(text=self.prompt_template.format(subject=block.text))
            for block in tool_input
        ]
        speech = GenerateSpeechTool()
        speech.generator_plugin_config = {
            "voice_id": VOICE_ID 
        }

        return speech.run(modified_inputs,context)

if __name__ == "__main__":
    tool = VoiceTool()
    with Steamship.temporary_workspace() as client:
        ToolREPL(tool).run_with_client(client=client, context=with_llm(llm=OpenAI(client=client)))
