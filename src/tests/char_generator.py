import json
from typing import Optional

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag
import os

GENERATOR_HANDLE = "together-ai-llm"
#GENERATOR_HANDLE = "lemonfox-streaming-llm"

with Steamship.temporary_workspace() as steamship:
    llama = steamship.use_plugin(GENERATOR_HANDLE,
                                 config={
                                     "api_key": os.environ['TOGETHER_KEY'],
                                     #"api_key": os.environ['LEMONFOX_KEY'],
                                     #"model": "zephyr-chat",
                                     "model": "NousResearch/Nous-Hermes-2-Yi-34B",
                                     "max_tokens": 2500,
                                     "temperature": 0.0,
                                     #"top_p": 0.9
                                 })
    file = File.create(
        steamship,
        blocks=[
            Block(
                text="""You are a creative role-play game level creator.""",
                tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                mime_type=MimeTypes.TXT,
            ),
            Block(
                text="""
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
