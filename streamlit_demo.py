import os
import streamlit as st
import time,requests
import numpy as np
from config.constants import PHQ_9

IMAGE_PATH="src/ML.jpeg"
IMAGE_PATH1="src/Psychology.png"
IMAGE_PATH2="src/prml.png"
IMAGE_PATH3="src/ml_whiteboard.png"
IMAGE_PATH4="src/splg.png"
IMAGE_PATH5="src/sml.png"
st.title("🌟 :blue[探索心理学与机器学习的奇妙交叉领域]🌟 ")
st.sidebar.markdown("# Main page 🎈")

st.markdown("&emsp;&emsp;欢迎来到🧠心大陆空间。我们致力于将心理学与机器学习结合，探索这两个领域的奇妙交叉点。\
    无论你是对心理学或机器学习感兴趣，还是想了解二者的共同之处，我们都为你准备了丰富的内容和深入的研究。\
    在这里，你将逐步了解将传统的统计机器学习方法、自然语言处理算法以及这些算法的实战演练，在学习的过程中你将会逐渐对这些算法有深刻的理解~")

st.image(IMAGE_PATH)
st.image(IMAGE_PATH1)
# 在正文区域创建一个块级列表
st.markdown("### 本文主要参考教材如下：")

options = [
    "🚀 [pattern recognition and machine learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)",
    "🔮[统计学习方法](https://github.com/fengdu78/lihang-code)",
    "⚡️[机器学习白板推导系列](https://space.bilibili.com/97068901)",
    "📘[speech and language processing](https://web.stanford.edu/~jurafsky/slp3/)",
    "📖[Reinforcement Learning (second edition)](http://incompleteideas.net/book/RLbook2020.pdf)"
]

for i,option in enumerate(options):
    st.markdown(f"##### {option}")
    
option_list=[]
sum_score=0
state=False
with st.expander(':orange[抑郁症筛查量表（PHQ-9）测评体验]'):
    st.markdown("###### &emsp;&emsp;请根据最近两个星期内您的实际感受，即以下症状在您的生活中出现的频率有多少？\
        选择一个与您的情况最符合的答案，不要花费太多时间去思考，根据第一印象做出判断，答案没有对错，真实反应自己的感受就好。")
    with st.form('my_form'):

        for index,question in enumerate(PHQ_9["Project"]):
            option_list.append(st.selectbox(":yellow[{}.{}]".format((index+1),question),PHQ_9["options"]))
            
        submitted = st.form_submit_button('Submit')
            
    score_list=[]
    if submitted:
        for opt in option_list:
                score_list.append(PHQ_9["score_map"][opt])
        sum_score=sum(score_list)
        st.success(f'''恭喜你完成填写，您的得分是`{sum_score}`!''', icon="✅")
        state=True
        #st.markdown(f'''##### 恭喜你成功完成填写,你的得分是`{sum_score}`''')
if state:
    with st.expander("请查看专业报告~"):    
            if sum_score>10:
                st.write("由于你的得分大于10，所以你有一点抑郁情绪")
            else:
                st.write("你很健康噢")
button_option_list=[]
                
            
            
if __name__=="__main__":
    pass

