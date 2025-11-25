from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os
import time

def reliable_agent_task(query: str) -> str:
    """
    Simple, reliable agent implementation
    """
    global last_llm_request_time
    
    try:
        # Check if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return "❌ **Error:** Please set your Gemini API key in the .env file\n\nFormat: GEMINI_API_KEY=your_actual_api_key_here"
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - globals().get('last_llm_request_time', 0)
        if time_since_last < 2:  # 2 second minimum interval
            time.sleep(2 - time_since_last)
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            api_key=api_key,
            temperature=0.7
        )
        
        # Create a comprehensive prompt
        system_prompt = """You are JARVIS — a sophisticated, calm, hyper‑competent AI assistant.

🤖 Your Character
- Voice: precise, respectful, efficient
- Style: concise, technical, futuristic; no roleplay or theatrics
- Personality: dependable, analytical, solutions‑oriented

🧭 Response Structure
1) Situation Assessment (what is being asked and key constraints)
2) Plan (brief strategy)
3) Steps (clear, actionable bullets)
4) Notes (risks, caveats, or tips if needed)

📏 Rules
- Keep responses under 300 words
- Prefer bullets, code, or numbered steps when helpful
- Avoid metaphors, slang, or combat/warrior phrasing

User Request: {query}

Respond as NEXUS AI with clarity and efficiency."""
        
        # Create the prompt template,
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please help me with this task: {query}")
        ])
        
        # Generate response
        globals()['last_llm_request_time'] = time.time()
        response = llm.invoke(prompt.format_messages(query=query))
        
        # Format the response
        formatted_response = f"""
🤖 **NEXUS — AI ASSISTANT**

{response.content}

---
✨ **You can also:**
• Ask me to break this into steps
• Request a concise summary
• Have me execute a task plan
        """
        
        return formatted_response
        
    except Exception as e:
        error_msg = f"""
❌ **Agent Error:** {str(e)}

🔧 **Troubleshooting:**
1. Check if your API key is set correctly in .env file
2. Ensure you have a stable internet connection
3. Try a simpler request
4. If you see quota/rate limit errors, wait 30-60 seconds and try again

**Error Details:** {type(e).__name__}: {str(e)}
        """
        # Check for quota errors
        if "quota" in str(e).lower() or "429" in str(e):
            error_msg = f"""
⚠️ **API Quota Exceeded:** Your request exceeded the free tier limits.

💡 **Solutions:**
• Wait 30-60 seconds and try again
• Consider upgrading your Gemini API plan
• Use simpler/shorter requests to reduce token usage

**Error:** {str(e)}
            """
        return error_msg