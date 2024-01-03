import json
from typing import Optional

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag
import os

GENERATOR_HANDLE = "together-ai-llm"

with Steamship.temporary_workspace() as steamship:
  llama = steamship.use_plugin(GENERATOR_HANDLE,
                               config={
                                   "api_key": os.environ['TOGETHER_KEY'],
                                   "model":
                                   "NousResearch/Nous-Hermes-Llama2-70b",
                                   "max_tokens": 1500,
                                   "temperature": 0.9,
                                   "top_p": 0.6
                               })
  file = File.create(
      steamship,
      blocks=[
          Block(
              text="""You are a creative character generator.""",
              tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
              mime_type=MimeTypes.TXT,
          ),
          Block(
              text=
              """Generate me a explicit and nsfw style anime female character, with the following format:
NAME:
\n
Char description (short description of type,age,gender etc.):
\n
Personality (personality,behaviour etc. described in detail):
\n
Appearance (description of character physical in detail as comma separated keywords):
\n
Seed chat message (short introduction message):
\n
""",
              tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
              mime_type=MimeTypes.TXT,
          ),
      ],
  )
  generate_task = llama.generate(input_file_id=file.id, )
  generate_task.wait()
  output = generate_task.output
  #print(output)
  print(output.blocks[0].text)
