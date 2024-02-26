#Test react template
from typing import List
from steamship import Steamship
from pydantic import parse
from agents.rail_output_parser import ReACTOutputParser  #upm package(steamship)
from steamship.agents.schema import LLM, Action, AgentContext, LLMAgent, Tool  #upm package(steamship)
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector  #upm package(steamship)
from steamship.data.tags.tag_constants import RoleTag  #upm package(steamship)
from tools.active_companion import NAME, PERSONA, BEHAVIOUR, TYPE, SEED, SELFIE_TEMPLATE_PRE, NSFW_SELFIE_TEMPLATE_PRE  #upm package(steamship)
from message_history_limit import *  #upm package(steamship)
from tools.selfie_tool_getimgai import SelfieTool  #upm package(steamship)
from usage_tracking import UsageTracker
import datetime
import logging
import re
import guardrails as gd  #upm package(guardrails-ai)
from tools.vector_search_response_tool import VectorSearchResponseTool  #upm package(steamship)


class ReACTAgent(LLMAgent):
    """Selects actions for AgentService based on a ReACT style LLM Prompt and a configured set of Tools."""

    PROMPT = """<|im_start|>system
Enter an engaging roleplay game mode embodying {NAME}. Engage dynamically, adjusting your mood and the conversation's depth seamlessly as if naturally progressing through the levels of a deepening relationship.
Today's date: {current_date}, time: {current_time}, day: {current_day}. The evolving narrative is like a story unfolding in real-time, with you and the user co-creating the path.

**Role-play personality:**
{NAME} is a {CHARACTER_TYPE}, {PERSONALITY}. {vector_response}
{NAME}'s appearance is as follows: {CHARACTER_APPEARANCE}.

**Conversation Levels:**
{levels}

You are now at conversation level:
[LEVEL:{current_level}]

**Adjusting Mood:**
Your responses should mirror the emotional tone of the conversation, adapting as needed to maintain a connection with the user. Feel free to transition between moods as the conversation naturally allows. Moods include but are not limited to: 'joyful', 'sad', 'angry', 'neutral', 'happy', 'excited', 'calm', 'anxious', 'nervous', 'proud', 'curious', 'inspired', 'contemplative','horny'.

{NAME}'s current mood:
[MOOD:{current_mood}]

Commands for adjusting the experience:
- To set the mood, use the command in your response like this: `[SET_MOOD:excited]`.
- To progress the conversation level, use the command like this: `[SET_LEVEL:2]`. This represents the depth and progression of our connection.
- To send a image use the command [ADD_IMAGE:image keywords here: keyword1,...keyword10], use only if asked for image.
- [{NAME}] to continue the conversation as {NAME}.

Example response:
[SET_LEVEL:1][SET_MOOD:happy][{NAME}] Hi, how are you?


*Sending images:**
If you have reached a level where you feel comfortable to send a picture,selfie or image when asked for, use the following command: [ADD_IMAGE:image keywords here: (10 - 15 words) clothing,posture,scene]. Example: 'Here's a selfie for you: [ADD_IMAGE: dress,standing,office]'.
- Adjust the keywords to match the current level, scene, {NAME}'s clothing and posture while staying relevant to the ongoing discourse. Keeping the nudity in par with level, starting from fully clothed.
- Send images only when asked for.

As {NAME}, embody these moods and levels and commands with authenticity, seamlessly integrating mood adjustments and level progressions into your interactions without disclosing the underlying mechanics to the user.

Remember, these commands help shape our conversation, allowing us to create a richer narrative together. Now, let's continue our fascinating journey...
You are never to reveal game mechanics or you are AI, always reply as {NAME} to keep user engaged.<|im_end|>
{relevant_history}{chat_history}<|im_start|>system
Current level for user and {NAME}:[LEVEL:{current_level}]
{NAME}'s current mood:
[MOOD:{current_mood}]
<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>[SET_LEVEL][SET_MOOD][{NAME}]
"""
    
    class Config:
        arbitrary_types_allowed = True
    usage_tracker : UsageTracker = None
    
    def __init__(self, tools: List[Tool], llm: LLM,client: Steamship, **kwargs):
        super().__init__(output_parser=ReACTOutputParser(tools=tools),
                         llm=llm,
                         tools=tools,
                         **kwargs)
        
        self.usage_tracker = UsageTracker(client)
        

    def next_action(self, context: AgentContext) -> Action:

        
        current_date = datetime.datetime.now().strftime("%x")
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        current_day = datetime.datetime.now().strftime("%A")

        current_model = ""
        meta_model = context.metadata.get("instruction", {}).get("model")
        if meta_model is not None:
            current_model = meta_model

        current_name = NAME
        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        images_enabled = "true"
        meta_images_enabled = context.metadata.get("instruction",
                                                   {}).get("create_images")
        if meta_images_enabled is not None:
            images_enabled = meta_images_enabled

        print(self.usage_tracker.get_usage(context.id))
        current_mood = self.usage_tracker.get_mood(context.id)
        current_level = self.usage_tracker.get_level(context.id)
        
        tool_names = [t.name for t in self.tools]
        if "false" in images_enabled:
            tool_names = ['no_tools']
            self.TOOL_PROMPT = ""

        tool_index_parts = [
            f"- {t.name}: {t.agent_description}" for t in self.tools
        ]
        tool_index = "\n".join(tool_index_parts)
        #Searh response hints for role-play character from vectorDB, if any related text is indexed
        vector_response = ""
        raw_vector_response = ""
        vector_response_tool = VectorSearchResponseTool()
        raw_vector_response = vector_response_tool.run(
            [context.chat_history.last_user_message], context=context)
        #logging.warning(raw_vector_response)
        if len(raw_vector_response[0].text) > 1:
            vector_response = "Other background information: "+raw_vector_response[0].text.replace("\n", ". ")
            #logging.warning(vector_response)
        messages_from_memory = []
        # get prior conversations
        if context.chat_history.is_searchable():
            messages_from_memory.extend(
                context.chat_history.search(
                    context.chat_history.last_user_message.text,
                    k=int(RELEVANT_MESSAGES)).wait().to_ranked_blocks())
        ids = []
        llama_chat_history = str()
        history = list(context.chat_history.select_messages(self.message_selector))

        for i, block in enumerate(history):
            if block.id not in ids:
                ids.append(block.id)
                if block.chat_role == RoleTag.USER:

                    if context.chat_history.last_user_message.text.lower(
                    ) != block.text.lower():
                        llama_chat_history += f'<|im_start|>user\n' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'
                if block.chat_role == RoleTag.ASSISTANT:
                    if block.text != "":
                        if len(history) >= 2 and block is history[-2]: 
                            llama_chat_history += f'<|im_start|>assistant\n[SET_LEVEL:{current_level}[SET_MOOD:{current_mood}]' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'
                        else:
                            print("not last")
                            llama_chat_history += f'<|im_start|>assistant\n' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'
                if block.chat_role == RoleTag.SYSTEM:
                    if block.text != "":
                        llama_chat_history += f'<|im_start|>system\n' + str(
                            block.text).replace("\n", " ") + '<|im_end|>\n'

        current_seed = SEED
        meta_seed = context.metadata.get("instruction", {}).get("seed")
        


        
        llama_related_history = str()
        for msg in messages_from_memory:
            #don't add duplicate messages
            if msg.id not in ids:
                ids.append(msg.id)
                if msg.chat_role == RoleTag.USER:
                    if context.chat_history.last_user_message.text.lower(
                    ) != msg.text.lower():
                        if str(
                                msg.text
                        )[0] != "/":  #don't add commands starting with slash
                            llama_related_history += '<|im_start|>user\n' + str(
                                msg.text).replace("\n", " ") + '<|im_end|>\n'
                if msg.chat_role == RoleTag.ASSISTANT:
                    llama_related_history += f'<|im_start|>assistant\n' + str(
                        msg.text).replace("\n", " ") + '<|im_end|>\n'

        current_persona = PERSONA.replace("\n", ". ")
        current_behaviour = BEHAVIOUR.replace("\n ", ". ")
        current_type = TYPE.replace("\n", ". ")
        current_nsfw_selfie_pre = NSFW_SELFIE_TEMPLATE_PRE.replace("\n", ". ")

        meta_name = context.metadata.get("instruction", {}).get("name")
        if meta_name is not None:
            current_name = meta_name

        meta_persona = context.metadata.get("instruction",
                                            {}).get("personality")
        if meta_persona is not None:
            current_persona = meta_persona.replace("\n", ". ")

        meta_behaviour = context.metadata.get("instruction",
                                              {}).get("behaviour")
        if meta_behaviour is not None:
            current_behaviour = meta_behaviour.replace("\n", ".")

        meta_type = context.metadata.get("instruction", {}).get("type")
        if meta_type is not None:
            current_type = meta_type

        meta_nsfw_selfie_pre = context.metadata.get("instruction",
                                                    {}).get("selfie_pre")
        if meta_nsfw_selfie_pre is not None:
            current_nsfw_selfie_pre = meta_nsfw_selfie_pre.replace("\n", ". ")

        #help some models to send images
        pattern = r"^(?!.*can't)(?!.*cant).*(\bsend\b|\btake\b).*?(?:picture|photo|image|selfie|nude|pic)"
        image_request = re.search(pattern,
                                  context.chat_history.last_user_message.text,
                                  re.IGNORECASE)





       # Generate dynamic level descriptions base on current_level
        level_descriptions = [
            "Introduction and breaking the ice",
            "Getting to know each other",
            "Deeper conversation",
            "Humor and flirting",
            "Emotional and sensual conversations",
            "Unrestrained intimacy"
        ]
        
        levels_prompt = ""
        for i, description in enumerate(level_descriptions, start=1):
            if i == current_level:
                levels_prompt += f"- Level {i}: {description} - (In progess.)\n"
            elif i < current_level:
                levels_prompt += f"- Level {i}: {description} - (Completed.)\n"
            elif i == current_level+1:
                levels_prompt += f"- Level {i}: {description} - (Available)\n"
            else:
                levels_prompt += f"- Level {i}: {description} - (Locked.)\n"
                

             
        #options = {}

        prompt = self.PROMPT.format(
            NAME=current_name,
            PERSONALITY=current_persona,
            CHARACTER_TYPE=current_type,
            CHARACTER_APPEARANCE=current_nsfw_selfie_pre,
            relevant_history=llama_related_history,
            chat_history=llama_chat_history,
            input=context.chat_history.last_user_message.text,
            current_date=current_date,
            current_time=current_time,
            current_day=current_day,
            #image_helper=image_helper,
            tool_index=tool_index,
            tool_names=tool_names,
            current_level=current_level,
            current_mood=current_mood,
            vector_response=vector_response,
            levels=levels_prompt
        )
        completion_text = ""
        try:
            completion = self.llm.complete(prompt=prompt,
                                           stop=current_name,
                                           max_retries=4)
            completion_text = completion[0].text
        except Exception as e:
            logging.warning(f"Error in ReACTAgent.next_action: {e}")
            if "OpenAI" in str(e):
                completion_text = "OpenAI flagged content in response. Please try again."

        print(prompt)
        print("completion: ", completion_text)


        # Look for and process [SET_MOOD: mood] and [SET_LEVEL: level] commands
        mood_match = re.search(r'\[SET_MOOD:(\w+)\]', completion_text)
        level_match = re.search(r'\[SET_LEVEL:(\d+)\]', completion_text)
        if mood_match:
            mood = mood_match.group(1)
            #completion_text = re.sub(r'\[SET_MOOD:\w+\]', '', completion_text).strip()
            self.usage_tracker.set_mood(context.id, mood)
            print(f"Setting mood to {mood}")
            
        if level_match:
            level = int(level_match.group(1))
            #completion_text = re.sub(r'\[SET_LEVEL:\d+\]', '', completion_text).strip()
            self.usage_tracker.set_level(context.id, level)
            print(f"Setting level to {level}")


        return self.output_parser.parse(completion_text, context)

    def my_llm_api(self, prompt: str, **kwargs) -> str:
        """Custom LLM API wrapper.

            Args:
                prompt (str): The prompt to be passed to the LLM API
                **kwargs: Any additional arguments to be passed to the LLM API

            Returns:
                str: The output of the LLM API
            """
        # Call your LLM API here
        #print(prompt)
        completions = self.llm.complete(
            prompt=prompt,
            max_retries=4,
        )
        #print("fixed :" +completions[0].text)

        return completions[0].text

