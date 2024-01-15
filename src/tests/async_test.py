import requests
import os
url = "https://mpoikkilehto.steamship.run/2t4yqcrtvjsdewrcfdp2xxk6h1y-0510a6aa57124621a644029046cc025a/2t4yqcrtvjsdewrcfdp2xxk6h1y-c065ba4882984f5e87779117ee8b51bf/async_prompt"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.environ["STEAMSHIP_API_KEY"]}'
}
data = {
    'prompt': 'VALUE',
    'context_id': 'VALUE',
    'kwargs': 'VALUE'
}
response = requests.post(url, json=data, headers=headers)
print(response)
print(response.text)