import requests
import os
from steamship import Steamship, Block, File
import json
url = "https://mpoikkilehto.steamship.run/space-f44596945d4245b4c7035d8c3bb3b6eb/backend-test-bot-b33119042929c5070def2ceba15d7836/async_prompt"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.environ["STEAMSHIP_API_KEY"]}'
}
data = {
    'prompt': 'send me a selfie',
}
response = requests.post(url, json=data, headers=headers)

response_data = response.json()
task_data = response_data.get("task", {})
task_request_id = task_data.get("requestId")

file_data = response_data.get("file", {})
file_id = file_data.get('id')
# Print the task_request_id and file_id
print(task_request_id, " : ", file_id, "\n\n\n")
if task_data.get("state") == "failed":
    raise Exception(f"Exception from server: {json.dumps(response)}")
chat_file_id = file_data.get("id")
request_id = task_data.get("requestId")

def stream_chat(response, access_token, stream_timeout=300, format="markdown"):
    if "status" in response and response["status"]["state"] == "failed":
        raise Exception(f"Exception from server: {json.dumps(response)}")

    chat_file_id = response["file"]["id"]
    request_id = response["task"]["requestId"]

    query_args = {
        "requestId": request_id,
        "timeoutSeconds": 300,
    }

    query_string = "&".join([f"{key}={value}" for key, value in query_args.items()])
    url = f"https://api.steamship.com/api/v1/file/{chat_file_id}/stream?{query_string}"
    print("SSE URL:",url)
    sse_url = url  # SSE stream URL

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "text/event-stream",
    }
    client = Steamship(api_key=access_token, workspace="space-f44596945d4245b4c7035d8c3bb3b6eb")
    with requests.get(sse_url, headers=headers, stream=True) as response:
        for event in response.iter_lines():
            if event is None:
                # Stream has ended, exit the loop
                break
            event_data = event.decode("utf-8")
            print((event_data))
            if event_data.startswith("data:"):
                data_str = event_data[len("data:"):]
                data_dict = json.loads(data_str)
                block_created_data = data_dict.get("blockCreated", {})
                request_id = block_created_data.get("requestId")
                block_id = block_created_data.get("blockId")
                mimeType = block_created_data.get("mimeType")
                created_at = block_created_data.get("createdAt")
                block=Block.get(client=client, _id=block_id)
                print(block.text,flush=True)
                print(block.mime_type)
                if block.mime_type == 'image/png':
                    #File.get()
                    print(block)
                    print(f"https://api.steamship.com/api/v1/block/{block.id}/raw")
                    #file = File.get(client=client, _id=block.id)
                #print(block)

stream_chat(response_data,os.environ["STEAMSHIP_API_KEY"])

