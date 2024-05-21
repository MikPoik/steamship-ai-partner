# Steamship AI Companion

## AI companion for next13-ai-companion frontend or as standalone bot

- vector database memory for learning, customizable personality.
- VectorDB learning configured, data can be added with api.
- Uses indexed data to create personal responses.
- Remembers details about user to provide personal experience
- Selfi and voice tools
- Keeps track of time and date.
- Keeps conversation history context in vector memory.
- Send telegram OGG voice messages with Transloadit-api (no autoplaying messages..)
- MoodTool to trigger different moods by keywords.
- Cost tracking by generated voice or tokens
- Telegram invoicing with Stripe
- User balance tracking
- Multiple image generation tools
- alternative llm's

## Installation

Follow Steamships install instructions for environment setup or use Replit python template.
Run ```pip install -r requirements.txt``` or ```poetry install```

## Customize
Companion templates in "companions" folder and active companion is set in "tools/active_companion.py"
## Running
Run locally with:
```python src/api.py``` or ```ship run local --no-ngrok```




