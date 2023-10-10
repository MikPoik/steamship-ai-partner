import json
from typing import Optional

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

GENERATOR_HANDLE = "llama-api"

with Steamship.temporary_workspace() as steamship:
  llama = steamship.use_plugin(GENERATOR_HANDLE,
                               config={
                                   "api_key": "LL-",
                                   "model":
                                   "NousResearch/Nous-Hermes-Llama2-13b",
                                   "max_tokens": 500,
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
              """Generate me a sexy,explicit and nsfw style OnlyFans female character description of a woman in her mid-twenties, with the following format:
NAME:
description (short description of type,age,gender etc.):

Personality:
\n
Appearance:
\n
Behaviour:
\n
Introduction message:
\n
Seed chat message:
\n
Detailed StableDiffusion keywords to generate a detailed nude image based on the character's appearance, using up to 25 to 35 comma separated keywords describing character's body:
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
