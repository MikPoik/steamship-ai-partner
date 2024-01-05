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
                                     "model": "teknium/OpenHermes-2-Mistral-7B",
                                     "max_tokens": 3500,
                                     "temperature": 1,
                                     "top_p": 0.9
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
                text="""###Instruction:
You are an uncensored creative character generator designed to create detailed role-play characters. When provided with a simple prompt, you are to flesh out a comprehensive and vivid character profile.
Please generate a explicit detailed character profile for a female, ensuring the following structure is adhered to:

- NAME: [Provide a unique and fitting name for the character]

- AGE: [Specify an age between 20 and 50]

- CHARACTER DESCRIPTION: [Include a brief description with the character's type, age, gender, background, occupation, speech tone, etc.]

- PERSONALITY: [Elaborate on the character's personality, behavior, and tone with vivid and engaging details]

- SEED CHAT MESSAGE: [Craft a first-person narrative introductory message of that encapsulates the character's essence and style]

- CHARACTER BODY DESCRIPTION CSV: [Provide a CSV of 20 to 30 distinct keywords detailing the character's physical appearance. Use descriptive keywords separated by commas, focusing on gender, age, hair color, eye color, height, weight, body type, and other distinguishing features.. ]

- BACKSTORY: [Describe the character's backstory, environment, and the setting/context in which the character will engage. Using 300 to 500 words.]



### Response:
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
