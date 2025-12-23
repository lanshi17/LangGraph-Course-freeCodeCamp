from typing import List, Optional, Any
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import SecretStr

import os

load_dotenv()

# 获取API密钥并进行类型转换
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key is None:
    raise ValueError("ANTHROPIC_API_KEY 环境变量未设置")

# 安全地获取BASE_URL，不直接打印
base_url = os.getenv("BASE_URL")
if base_url:
    print(f"Connecting to: {base_url}")