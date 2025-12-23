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

@dataclass
class AgentState:
    messages: List[BaseMessage]

# 安全地获取超时值，添加异常处理
try:
    timeout_value = int(os.getenv("ANTHROPIC_TIMEOUT", "60"))
except ValueError:
    timeout_value = 60  # 默认值

llm = ChatAnthropic(
    model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    api_key=SecretStr(api_key),
    base_url=base_url, 
    timeout=timeout_value,
    max_retries=2, 
    stop=None,
)

def process(state: AgentState) -> AgentState:
    try:
        response = llm.invoke(state.messages)
        print(f"AI:{response.content}")
        return state
    except Exception as e:
        print(f"LLM调用失败: {e}")
        # 返回原始状态，不修改消息列表
        return state

# 创建 Graph 流程
def create_agent_graph():
    """创建一个简单的对话 Agent Graph"""
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("process", process)
    
    # 添加边：从 START 连接到 agent 节点，从 agent 节点连接到 END
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    
    # 编译图
    return graph.compile()


# 示例：使用 Agent
def example(agent: Any) -> None:
    """演示如何使用 Agent Graph"""
    user_input = input("Enter:")
    while user_input!="exit":
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input=input("Enter:")

# 绘制图的流程
def visualize_graph(graph: Any)->None:
    """绘制并保存 Agent Graph 的 Mermaid 图"""
    try:
        # 生成 Mermaid 图
        mermaid_png = graph.get_graph().draw_mermaid_png()
        
        # 保存为 PNG 文件
        output_path = "agent_graph.png"
        try:
            with open(output_path, "wb") as f:
                f.write(mermaid_png)
        except IOError as e:
            print(f"✗ 保存文件失败: {e}")
            return None
        
        print(f"✓ Graph 已绘制并保存到: {output_path}")
        
        # 打印 Mermaid 代码
        mermaid_code = graph.get_graph().draw_mermaid()
        print("\n=== Mermaid 图代码 ===")
        print(mermaid_code)
        
        return mermaid_png
    except Exception as e:
        print(f"✗ 绘制图失败: {e}")
        print("  请确保已安装: pip install pillow graphviz")
        return None

if __name__ == "__main__":
    # 运行示例
    graph = create_agent_graph()
    example(graph)
    visualize_graph(graph)