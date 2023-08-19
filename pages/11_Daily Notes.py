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
    st.markdown("&emsp;&emsp;如果是下面的多轮对话数据，我们可以引入对话角色的标识，比如:blue['求助者：']和:blue['支持者：']，当然，换成:blue['Human：']与:blue['Assistant：']也完全没问题。")
    multi_turn_conversation='''  "求助者：最近总是和妈妈闹矛盾，但是又不知道该怎么办，能帮我想想办法吗？",\n
  "支持者：我了解你的情况，跟亲人之间经常会产生矛盾是很常见的现象。你不妨试试和妈妈沟通一下，平静地提出自己的疑惑和不满，看看能否解决矛盾。",\n
  "求助者：但是每次我和妈妈说话，总会起争端，她总是让我感觉她不信任我，我该怎么办呢？",\n
  "支持者：听起来你和妈妈之间的交流很困难，你可以试试换个方式和她沟通，比如写信或者找一个更加中立的人一起协调谈话，让大家都有更好的表达机会。",\n
  "求助者：我特别讨厌和她吵架，可是我有时候就是自制力不够，很难抑制自己的情绪。",\n
  "支持者：青春期的年轻人情绪波动很大很正常，但是你可以试试找些方法来缓解情绪，比如听听音乐、看看书等等，使自己情绪更稳定。"'''
    st.code(multi_turn_conversation,language="shell")
    st.markdown("&emsp;&emsp;最后将多轮对话数据通过'</s>'拼接得到一条完整的训练样本。")
    code_samples='''"input1</s>target1</s>input2</s>target2</s>...inputn</s>target</s>"'''
    st.code(code_samples,language="shell")
    st.markdown("&emsp;&emsp;将数据准备好以后，那么该采用什么样的损失函数来训练模型呢？我们可以用Transformers默认的:blue[AutoModelForCausalLm]进行训练，无需改写损失函数。\
        这样的特点是，在训练时模型会不断地根据上文来预测下一个token，即属于人类说话的部分也会被预测，这一部分的损失不会被忽略。我们也可以改写损失函数，将属于人类说话部分的损失忽略(Mask掉)，即只预测模型回答的那一部分。两种方式都是可行的。")
    st.markdown("## :blue[]")

def page_inference_speed():
    st.title("模型加速推理")
    st.markdown("## :blue[fastllm模型加速]")
    # 在关于页面中添加内容

def page_paper_reading():
    st.title("联系我们")
    # 在联系我们页面中添加内容
    
def page_others():
    st.title("其他想法")
# 定义侧边栏选项

pages = {
    "大语言模型训练": page_llm_training,
    "加速推理方案": page_inference_speed,
    "论文解读": page_paper_reading,
    "其他想法": page_others
}

# 添加侧边栏菜单
selection = st.sidebar.radio("系列精选", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()