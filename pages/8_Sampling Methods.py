import streamlit as st
from chapters.statistical_machinelearning.chapter9_sampling import Markov_Chain_Monte_Carlo


pages={
    "Markov Chain Monte Carlo": Markov_Chain_Monte_Carlo,
}
# 添加侧边栏菜单
selection = st.sidebar.radio("文章分类", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()