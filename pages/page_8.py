import streamlit as st

def page_llm_training():
    st.title("大语言模型训练")
    st.markdown("")
    st.image("src/Instruction_tuning.gif")
    # 在首页中添加内容

def page_inference_speed():
    st.title("模型加速推理")
    st.markdown("## :blue[fastllm模型加速]")
    # 在关于页面中添加内容

def page_paper_reading():
    st.title("联系我们")
    # 在联系我们页面中添加内容

# 定义侧边栏选项
pages = {
    "大语言模型训练": page_llm_training,
    "加速推理方案": page_inference_speed,
    "论文解读": page_paper_reading
}

# 添加侧边栏菜单
selection = st.sidebar.radio("系列精选", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()