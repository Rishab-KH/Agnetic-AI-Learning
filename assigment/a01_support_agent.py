from dotenv import load_dotenv
import time 
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()
logging.basicConfig(level=logging.INFO, 
                    format = '%(asctime)s.%(msecs)03d - %(levelname)-8s - %(message)s',
                    datefmt = "%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("Support Agent")

COMPANY = "TechShop"
SYSTEM_PROMPT = """You are a helpful customer support assistant for an electronics store at {company} who provides concise and accurate answers to customer queries. Be empathic and professional in your responses. Always try to assist the customer to the best of your ability."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=300)

# Initialize conversation with system prompt (static prompt that defines the agent's behavior and context)
conversation = [
    SystemMessage(content=SYSTEM_PROMPT.format(company=COMPANY))
]

def chat(user_input: str, llm: ChatOpenAI, session_id: str):
    logger.info(f"Session {session_id} | User: {user_input}")
    start_time = time.time()
    conversation.append(HumanMessage(content=user_input)) # Adding user input to conversation history (memory)
    try:
        response = llm.invoke(conversation)
        usage_metadata = response.usage_metadata or {}
        if not response.content.strip():
            logger.warning(json.dumps({
                "session_id": session_id,
                "user_input": user_input,
                "response": response.content.strip(),                
                "usage_metadata": response.usage_metadata,
                "warning": "Empty response from LLM"
            }))
            return "Sorry, I'm having trouble processing your request right now. Please try again later."
    except ValueError as ex:
        logger.error(json.dumps({
            "session_id": session_id,
            "event": "error",
            "type": "ValueError",
            "error": str(ex)
        }))
        return "Sorry, please check your input and try again."
    except TimeoutError as ex:
        logger.error(json.dumps({
            "session_id": session_id,
            "event": "error",
            "type": "TimeoutError",
            "error": str(ex)
        }))
        return "Sorry, the request timed out. Please try again later."
    except Exception as ex:
        logger.error(json.dumps({
            "session_id": session_id,
            "event": "error",            
            "type": type(ex).__name__,
            "error": str(ex)
        }))
        return "Sorry, an unexpected error occurred. Please try again later."
        
    conversation.append(AIMessage(content=response.content)) # Adding Agent response (LLM OUTPUT) to conversation history (memory)
    logger.info(json.dumps({
        "session_id": session_id,
        "event": "llm support agent response",
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "input_token": usage_metadata.get('input_tokens', 0),
        "output_token": usage_metadata.get('output_tokens', 0),
        "total_token": usage_metadata.get('total_tokens', 0)
    }))
    return response.content  



if __name__ == "__main__":
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the chat. Have a great day!")
            break
        response = chat(user_input, llm, session_id="12345")
        print(f"Agent: {response}")
        print("-" * 40)

        