"""Tool for generating audio"""
from steamship import Steamship  #upm package(steamship)
from steamship.agents.llms import OpenAI  #upm package(steamship)
from steamship.agents.tools.speech_generation import GenerateSpeechTool  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task  #upm package(steamship)
from steamship.agents.schema import AgentContext
from steamship.agents.schema import AgentContext, Tool  #upm package(steamship)
import logging
import re
#from tools.active_companion import VOICE_ID #upm package(steamship)


class CoquiTool(Tool):
  """Tool to generate audio from text."""

  name: str = "GenerateSpokenAudio"
  human_description: str = "Generates spoken audio from text."
  agent_description: str = (
      "Use this tool to generate spoken audio from text, the input should be a plain text string containing the "
      "content to be spoken.")

  prompt_template = ("{subject}")
  generator_plugin_handle: str = "coqui-tts"
  generator_plugin_config: dict = {
      "coqui_api_key":
      "YWEmkTs4lWiipl2BFOdQqRquJaskhXjRQNTWuEdEMSPHwvgjsrgNLsr2esH1hGZd",
      "language": "en",
      "speed": 1.2
  }

  def run(self, tool_input: List[Block],
          context: AgentContext) -> Union[List[Block], Task[Any]]:

    meta_voice_id = context.metadata.get("instruction", {}).get("voice_id")
    if meta_voice_id is not None:
      if meta_voice_id != "none":
        self.generator_plugin_config["voice_id"] = meta_voice_id

    generator = context.client.use_plugin(self.generator_plugin_handle,
                                          config=self.generator_plugin_config)
    text_input = ""
    for block in tool_input:
      if not block.is_text():
        continue
      text_input += block.text

    pattern = r'\*([^*]+)\*'  #remove *gesture* from text
    text_input = re.sub(pattern, '', text_input)
    #logging.warning("coqui input: " + text_input)
    task = generator.generate(text=text_input,
                              make_output_public=True,
                              append_output_to_file=True)
    task.wait()
    blocks = task.output.blocks
    output_blocks = []
    for block in blocks:
      output_blocks.append(block)
    return output_blocks


if __name__ == "__main__":
  tool = CoquiTool()
  with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client,
                                   context=with_llm(llm=OpenAI(client=client)))
