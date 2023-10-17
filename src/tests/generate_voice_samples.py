from steamship import Block, File, MimeTypes, Steamship, Task
from time import sleep


def main():
  client = Steamship(workspace="coqui-sample-ws")

  generator = client.use_plugin("coquidev",
                                config={
                                    "coqui_api_key":
                                    "",
                                    "voice_id":
                                    "c791b5b5-0558-42b8-bb0b-602ac5efc0b9"
                                })

  text = "Want to hear a song?"
  task = generator.generate(
      text=text,
      append_output_to_file=True,
      make_output_public=True,
  )

  task.wait()
  url = f"https://api.steamship.com/api/v1/block/{task.output.blocks[0].id}/raw"

  print(url)
  return task.output.blocks


if __name__ == "__main__":
  main()
