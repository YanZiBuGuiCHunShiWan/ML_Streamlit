import os
import streamlit as st

IMAGE_PATH="src/ML.jpeg"
IMAGE_PATH1="src/Instruction_tuning.gif"
IMAGE_PATH2="src/prml.png"
st.title(":blue[机器学习理论与实践] 🎈")
st.sidebar.markdown("# Main page 🎈")
st.title(":blue[Welcome to my space~]")
st.image(IMAGE_PATH)
st.image(IMAGE_PATH1)
# 在正文区域创建一个块级列表
st.markdown("## 本文主要参考教材如下：")

options = [
    "[pattern recognition and machine learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)",
    "[统计学习方法](https://github.com/fengdu78/lihang-code)",
    "[机器学习白板推导系列](https://space.bilibili.com/97068901)",
    "[speech and language processing](https://web.stanford.edu/~jurafsky/slp3/)"
]

for i,option in enumerate(options):
    
    st.markdown(f"- {option}")
    if i==0:
        st.image(IMAGE_PATH2)
    
    
    
    
button_home = st.button("首页")
button_products = st.button("产品")
button_solutions = st.button("解决方案")

# 根据按钮点击情况显示内容
if button_home:
    st.header("欢迎来到首页！")
    # 显示首页内容...

if button_products:
    st.header("这是我们的产品页面！")
    # 显示产品页面内容...

if button_solutions:
    st.header("这是我们的解决方案页面！")
if __name__=="__main__":
    pass

