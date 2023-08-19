import streamlit as st

def page_causal_inference():
    st.title("因果推断介绍")
    st.markdown("")
    # 在首页中添加内容

def page_causal_inference_basic():
    st.title("因果推断基础")
    st.markdown("## :blue[fastllm模型加速]")
    # 在关于页面中添加内容


# 定义侧边栏选项
pages = {
    "因果推断":page_causal_inference,
    "因果推断基础理论":page_causal_inference_basic
}

# 添加侧边栏菜单
selection = st.sidebar.radio("系列精选", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()