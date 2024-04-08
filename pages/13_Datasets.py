import streamlit as st
from chapters.datasets.datasets import Custom_Datasets

def paper_PsyQA():
    st.markdown("## :blue[对话数据集]")
    st.markdown("### :blue[[1.PsyQA: A Chinese Dataset for Generating Long Counseling Text for Mental Health Support](https://arxiv.org/abs/2106.01702)]")
    st.markdown("&emsp;&emsp;论文提出了中文心理领域的对话数据集PsyQA。数据是通过爬取[壹心理心理服务平台](http://www.xinli001.com/qa)后清洗得来的。数据标注的格式详见[此处](https://github.com/thu-coai/PsyQA/blob/main/PsyQA_example.json)。")
    st.markdown("##### :blue[Data cleaning]")
    st.markdown("&emsp;&emsp;数据清洗的规则如下：1.基于规则的方式移除个人信息、去除重复换行符、表情、网页链接、广告；2.为确保回复质量，只保留超过100字的回复；3.基于关键词过滤和主题无关的提问帖。")
    st.markdown("##### :blue[Strategy annotation]")
    st.markdown("&emsp;&emsp;作者认为（假设）帖子的回复是由策略序列遵循某种规律构成的，策略详情如下图：")
    st.image("src/paper-datasets-1-strategy.png")
    st.markdown("##### :blue[ Strategy Identification]")
    st.markdown("&emsp;&emsp;作者用Roberta训练了一个句子级别的策略分类器，如果有连续的句子$S_1, S_2, S_3, · · ·$，那么添加上分隔符得到$[CLS]S_1[SEP][CLS]S_2[SEP][CLS]S_3$作为模型的输入，然后取$[CLS]$位置的损失的平均作为整体的损失。\
        如果没有上下文只有单一的句子$S_1$,那么就只有一个$[CLS]$的损失。")
    st.markdown("&emsp;&emsp;作者分别比较了有上下文信息的模型和没有上下文信息的模型结果，在准确率上，有上下文信息为$74.81\%$，无上下文信息为$73.74\%$。")
    st.markdown("##### :blue[Answer Generation]")
    st.markdown("- 任务定义：给定一个三元组（问题$S_Q$，描述$S_D$，关键字集$K$）作为输入，其中$S_Q，S_D$都是句子，$K$最多由4个关键词组成，任务是生成一个由多个句子组成的咨询，可以提供有帮助的安慰和模仿心理健康顾问的建议。")
    st.markdown("&emsp;&emsp;首先预训练一个GPT2模型，将人类标注的strategy和Roberta标注的strategy混合，然后划分训练集、验证集、测试集。")
    st.markdown("&emsp;&emsp;:red[在微调时]，通过$prepending$即扩展其词汇表添加自定义的特殊令牌将不同的句子拼接起来，具体如下：\
         $[QUE]S_Q[DESC]S_D[KWD]K[ANS]$，其中$S_Q$是提问，$S_D$是问题的描述，$K$是关键词，这几个Token都会预先定义好；$[ANS]$即答案部分具体拼接如下：$[Strategy1]S_1[Strategy2]S_2[Strategy3]S_3 · · ·$")
    st.markdown("&emsp;&emsp;结果如下：")
    st.image("src/paper-datasets-2-result.png")
    st.markdown("&emsp;&emsp;表明在微调时加上句子的strategy标签能够显著提升模型的语义理解能力。注：因为在训练时的语料有prepend token，因此在模型预测时\
        也会输出prepend token，只是作者在提供案例时去掉了prepend token，并用不同颜色表明策略对应的回复内容。")
    st.markdown("&emsp;&emsp;局限性：作者的语料只针对于单轮对话数据，没用多轮对话数据，因此没法验证模型对上下文的理解能力如何。此外，受限于训练的策略分类器\
        准去率，模型在给语料预测策略时很可能出错，从而影响微调结果。")
    
    pass
pages={
    "PsyQA": paper_PsyQA,
    "A Multi-turn dialogue Generation Method Based on Markov Chain and LLM Interaction": Custom_Datasets
}
# 添加侧边栏菜单
selection = st.sidebar.radio("数据集", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()