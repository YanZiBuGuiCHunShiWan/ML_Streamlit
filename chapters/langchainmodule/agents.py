import streamlit as st


def Agents():
    st.markdown("## :blue[Agents]")
    st.markdown("&emsp;&emsp; 我们看一个langchain的简单案例，如下所示：")
    agent1_code='''
    from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model=llm_model)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
agent("What is the 25% of 300?")
    '''
    st.code(agent1_code,language="python")
    st.image("src/langchainsource/agents/agent1.png",caption="Fig.1:执行结果")
    st.image("src/langchainsource/agents/agent2.png",caption="Fig.2:create_prompt方法")
    st.markdown("&emsp;&emsp;我们基于ZeroShotAgent这一个类进行分析（libs/langchain/langchain/agents/mrkl/base.py），先看其类方法```create_prompt```如上图，从其代码可以分析个大致。\
        其中参数```format_instructions```默认是一个常量，我们可以ctrl+点击去看一下")
    st.markdown("&emsp;&emsp;下图是prompt的内容(libs/langchain/langchain/agents/mrkl/prompt.py)")
    agent2_code='''
    # flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    '''
    st.code(agent2_code,language="shell")