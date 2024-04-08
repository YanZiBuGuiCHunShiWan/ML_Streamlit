import streamlit as st
from chapters.statistical_machinelearning.chapter5_kernel_methods import Reproducing_kernel_hilbert_space

pages={
    "1.再生核希尔伯特空间": Reproducing_kernel_hilbert_space,
}
# 添加侧边栏菜单
selection = st.sidebar.radio("文章分类", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()