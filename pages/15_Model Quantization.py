import streamlit as st
from chapters.quantization.modelquantize import ModelQuantizationComprehend

def Part1_QuantizationIntroduction():
    st.markdown("### :blue[模型量化]")
    
    
    pass
pages={
    "模型量化介绍": ModelQuantizationComprehend,
}
# 添加侧边栏菜单
selection = st.sidebar.radio("数据集", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()