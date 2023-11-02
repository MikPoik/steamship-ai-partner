from transloadit import client  #upm package(pytransloadit)

# initialize the client
tl = client.Transloadit('', '')

# new assembly
assembly = tl.new_assembly()

# Step 1: Import audio file 1
assembly.add_step(
    'import1', '/http/import', {
        'url':
        'https://storage.googleapis.com/coqui-samples/XTTS_Personalities_Meet_the_Gang_Aaron%20Dreschner_3f627f80-b679-41a1-bd04-eacc924f12b0.wav'
    })

# Step 2: Import audio file 2
assembly.add_step(
    'import2', '/http/import', {
        'url':
        'https://storage.googleapis.com/coqui-samples/XTTS_Personalities_Meet_the_Gang_Abrahan%20Mack_3f627f80-b679-41a1-bd04-eacc924f12b0_.wav'
    })

# Step 3: Concatenate the imported audio files.
assembly.add_step(
    'concat', '/audio/concat', {
        "use": {
            "steps": [
                {
                    "name": "import1",
                    "as": "audio_1"
                },
                {
                    "name": "import2",
                    "as": "audio_2"
                },
            ]
        },
        "result": {
            "wav": True
        }
    })

# execute assembly
assembly_response = assembly.create(retries=5, wait=True)
print(assembly_response)
print(assembly_response.data.get('assembly_ssl_url'))

# get the concatenated audio file's url
concat_file = assembly_response.data['results']['concat'][0]['ssl_url']
print(concat_file)
