import streamlit as st
from chapters.papers.AI_agents import (Agents_prompt_engineering,
                                       Domain_specialization_as_the_key,
                                       Imitating_proerties_llm,
                                       ChainofThoughtPrompting,
                                       LLM_Powered_Autonomous_Agents)
pages={
    "大语言模型领域专业化": Domain_specialization_as_the_key,
    "提示工程": Agents_prompt_engineering,
    "Chain of Thought ":ChainofThoughtPrompting,
    "LLM Agents":LLM_Powered_Autonomous_Agents
}
# 添加侧边栏菜单
selection = st.sidebar.radio("文章分类", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()