from steamship import Steamship #upm package(steamship)
from steamship.utils.kv_store import KeyValueStore #upm package(steamship)
import json

if __name__ == "__main__":
    #used workspace
    client = Steamship(workspace="partner-ai-dev2-ws")
    
    kv_store = KeyValueStore(client, store_identifier="usage_tracking")
    item = kv_store.items()
    for id in range(len(item)):
        print( item[id])
        
    #create a json if needed
    items_json = json.dumps(item)
    #todo save json do local file?
    

    