import re

def contains_send_with_keywords(text:str):
    pattern = r'\b(?:send|picture|photo|image|selfie|nude|pic)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

text = "Please send me a selfie."
if contains_send_with_keywords(text):

    print("Found 'send' with one of the keywords.")
else:
    print("Not found.")
