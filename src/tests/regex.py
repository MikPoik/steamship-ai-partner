import re

def contains_send_with_keywords(text:str):
    pattern = r'\bsend\b.*?(?:picture|photo|image|selfie|nude|pic)'
    return bool(re.search(pattern, text, re.IGNORECASE))

text = "send this image"
if contains_send_with_keywords(text):

    print("Found 'send' with one of the keywords.")
else:
    print("Not found.")
