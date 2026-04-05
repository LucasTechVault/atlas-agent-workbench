from langchain_openai import ChatOpenAI
from app.config.settings import settings

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key
    )

# get_reasoning_llm
# get_tool_llm etc.