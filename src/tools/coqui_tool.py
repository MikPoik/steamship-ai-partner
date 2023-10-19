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
      "coqui_api_key": "",
      "language": "en",
      "speed": 1.0
  }

  def run(self, tool_input: List[Block],
          context: AgentContext) -> Union[List[Block], Task[Any]]:

    def process_and_call_tool(batch_text):
      generator = context.client.use_plugin(
          self.generator_plugin_handle,
          config=self.generator_plugin_config,
          version="0.0.3")
      task = generator.generate(text=batch_text,
                                make_output_public=True,
                                append_output_to_file=True)
      task.wait()
      return task.output.blocks

    output_blocks = []

    meta_voice_id = context.metadata.get("instruction", {}).get("voice_id")
    if meta_voice_id is not None:
      if meta_voice_id != "none":
        self.generator_plugin_config["voice_id"] = meta_voice_id

    text_input = ""
    for block in tool_input:
      if not block.is_text():
        continue
      text_input += block.text

    words = text_input.split()
    current_batch = []
    current_length = 0

    for word in words:
      word_length = len(word)
      if current_length + word_length + len(current_batch) > 250:
        # If adding the current word exceeds the batch size, process the current batch
        batch_text = ' '.join(current_batch)
        output_blocks.extend(process_and_call_tool(batch_text))
        current_batch = []
        current_length = 0

      current_batch.append(word)
      current_length += word_length

    # Process the last batch
    if current_batch:
      batch_text = ' '.join(current_batch)
      output_blocks.extend(process_and_call_tool(batch_text))

    return output_blocks


if __name__ == "__main__":
  tool = CoquiTool()
  with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client,
                                   context=with_llm(llm=OpenAI(client=client)))
