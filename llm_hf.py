import os
from typing import List, Any, Optional
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# 1. Load the .env file
load_dotenv()

# 2. Get your token from the .env
hf_token = os.getenv("HF_READ_SVT_TOKEN")

# 3. CRITICAL: Map it to the name LangChain requires
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
else:
    raise ValueError("Could not find HF_READ_SVT_TOKEN in .env file!")




# Using the powerful 7B model (runs in the cloud, saves your RAM!)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 3. Setup the Remote Engine
llm_engine = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="text-generation",
    max_new_tokens=1000,
    temperature=0,  # Slight temperature for more stable remote responses
    huggingfacehub_api_token= hf_token
)

# 4. Wrap it in ChatHuggingFace with the specific model_id
# This ensures the LLM knows how to format the <tool_call> XML tags correctly
llm = ChatHuggingFace(llm=llm_engine)

# llm_hf.py snippet
async def call_llm(
    messages: List[Any], 
    tools: Optional[List[Any]] = None,
):
    """
    Invokes the Hugging Face LLM using a message list.
    Supports asynchronous execution.
    """
    if tools:
        # Use .bind_tools() to attach the tool schemas to the model
        model_with_tools = llm.bind_tools(tools)
        return await model_with_tools.ainvoke(messages)

    return await llm.ainvoke(messages)