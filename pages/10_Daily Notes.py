import streamlit as st

def page_llm_training():
    st.title("大语言模型训练")
    st.markdown("")
    st.image("src/Instruction_tuning.gif")
    st.markdown("## :blue[训练详解]")
    # 在首页中添加内容
    st.markdown("&emsp;&emsp;Causal Language Model（因果语言模型）是一种基于序列的语言模型，它根据前面的上文生成下一个词或字符，该模型主要用于生成文本，具体的生成过程如上图所示。")
    st.markdown("&emsp;&emsp;给定一个单论对话的数据，输入是：:blue['给定一个英文句子，翻译成中文。\nI love to learn new things every day.\n']，回答是：:blue['我每天喜欢学习新事物。']。现在我们想训练一个因果语言模型完整这个问答任务，那么该如何建模呢？")
    st.markdown("&emsp;&emsp;如果直接把输入和回答拼接在一起变成':blue[给定一个英文句子，翻译成中文。 I love to learn new things every day.我每天喜欢学习新事物。]'让模型根据上文生成下一个token,那么存在两个问题：\
        1. :red[模型分不清输入和回答的部分]，2.:red[模型不知道何时结束]。")
    st.markdown("&emsp;&emsp;我们可以通过引入额外的标识符和'</s>'添加有效的区分信息。具体如下：")
    code_content=''' "Human: "+"给定一个英文句子，翻译成中文。\\nI love to learn new things every day.\\n"+"\\n</s>"+"<s>Assistant: "+"我每天喜欢学习新事物。"+"</s>" '''
    st.code(code_content,language="shell")
    st.markdown("## :blue[]")

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