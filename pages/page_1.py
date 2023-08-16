import streamlit as st
def page_home():
    st.title("首页")
    # 在首页中添加内容

def page_about():
    st.title("关于")
    # 在关于页面中添加内容

def page_contact():
    st.title("联系我们")
    # 在联系我们页面中添加内容

# 定义侧边栏选项
pages = {
    "首页": page_home,
    "关于": page_about,
    "联系我们": page_contact
}

# 添加侧边栏菜单
selection = st.sidebar.radio("导航菜单", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()