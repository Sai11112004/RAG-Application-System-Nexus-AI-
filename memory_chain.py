import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Rate limit variables
last_request_time = 0
MIN_REQUEST_INTERVAL = 2  # seconds between requests
MAX_RETRIES = 3           # retry attempts for API errors


# Initialize Gemini LLM
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Memory buffer
memory = ConversationBufferMemory(return_messages=True)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful AI assistant. Continue the conversation based on memory.

Conversation History:
{history}

Human: {input}
AI:
"""
)

chat_chain = LLMChain(llm=_llm, prompt=prompt_template, memory=memory)

def chat_with_memory(user_input: str) -> str:
    global last_request_time

    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - time_since_last)

    try:
        last_request_time = time.time()
        response = chat_chain.invoke({"input": user_input})
        return response["text"] if isinstance(response, dict) else response

    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return "⚠️ API quota exceeded. Please wait a moment and try again. This is due to free tier limits."
        else:
            return f"⚠️ Error: {str(e)}"
