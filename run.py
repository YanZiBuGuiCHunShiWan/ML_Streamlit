import streamlit as st
import pandas as pd
st.markdown('# Learning to Rank')
st.markdown('​在信息检索和推荐系统领域，排序问题始终是核心任务之一。从搜索引擎返回的网页列表，到电商平台为用户推荐的商品，排序算法无处不在。为了更智能、更个性化地进行排序，**Learning to Rank（学习排序）** 应运而生。Learning to Rank 的起源可以追溯到 2000 年代初期，随着机器学习在自然语言处理和信息检索中的广泛应用，人们逐渐意识到传统的基于规则或启发式的排序方法难以应对复杂的用户需求。2005 年，微软亚洲研究院发表了著名的 $\\text{RankNet}$（基于神经网络的排序学习模型），随后又推出 $\\text{LambdaRank }$和 $\\text{LambdaMART}$，这些工作开启了用监督学习方法直接优化排序的新时代。排序方法整体可分为Point-Wise,Pair-Wise,List-Wise三种，本文接下来讲按照顺序介绍文档排序场景下这三种方法的思想与具体细节。')
st.markdown('# 1.Point-Wise')
st.markdown('​Point-Wise Ranking 是学习排序的一类方法，它把排序任务视为 **回归或分类问题**。因此通常采样$\\text{BCE Loss}$或者$\\text{Focal Loss}$作为策略，以文档排序的场景为例，我们用BERT作为Cross Encoder捕获查询和文档间细粒度的语义交互，给定一个查询$query_i$、相关的文档$doc_i^+$（这里笔者假定只有一个相关文档，实际上可以有多个）和对应的$m$个候选文档$doc_{ij}^-,j=1,\ldots,m$，将$query_i$和对应的文档通过特殊符号$\\text{[CLS][SEP]}$拼接后作为BERT的输入，由可训练的线性层$\mathbf W$映射后再经过$\operatorname{sigmoid}$函数得到对应的分数$s_i$，对于正样本的得分$s_i$，应该越接近$1$越好，对于负样本得分$s_j$，应该越接近$0$越好。如下图：')
st.image('assets/image-20250701180409168.png')
st.markdown('​Point-Wise 把排序问题当作 **独立的回归或分类任务** 来做，预测每个样本的分值或概率。但排序真正关心的是 **文档之间的相对顺序**（比如NDCG、MAP、MRR 等），Point-Wise 并没有直接针对这些指标优化，因此即便模型预测的分值接近真实分值，也可能导致最终的排序顺序完全错误。此外，Point-Wise损失函数通常不能反映“局部排序错误”的严重程度，如把排名第$1$的文档得分预测稍低一些，导致其拍到了后几位，损失函数依然非常小，但是上线后用户体验和位置有关的衡量指标都很差。如果训练集中有大量负样本，模型可能只学会输出低分来降低损失，即便是类别加权的损失也难以将模型改进到正常水平。')
st.markdown('# 2.Pair-Wise')
st.markdown('## 2.1 RankNet& lambda Rank')
st.markdown('​$\mathrm{RankNet}$[[1]](https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf)的核心思想是使用 **成对比较（pairwise approach）** 来学习一个排序函数，该函数可以根据文档对$(doc_i,doc_j)$的相关性预测它们相对于查询$q$的排序顺序。$\mathrm{RankNet}$ 的损失函数可以表示为[[2]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)：')
st.latex(r'''\begin{align}C=\frac{1}{2}\left(1-S_{i j}\right) \sigma\left(s_{i}-s_{j}\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)\end{align}''')
st.markdown('​对于给定的查询$Q$，$S_{ij}\in\{−1,0,1\}$取值如下：$S_{ij}=1$：如果 $doc_i$ 比 $doc_j $更相关；$S_{ij}=0$：如果 $doc_i$ 和 $doc_j$ 相关性相同；$S_{ij}=−1$：如果 $doc_i$ 比更不相关。$s_i$ 和$s_j$分别表示文档$doc_i$和$doc_j$的相关性评分。$\sigma$是一个超参数，用于缩放$s_i-s_j$​的值。')
st.markdown('​当$S_{ij}=1$时有：')
st.latex(r'''\begin{aligned}C=\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)\end{aligned}''')
st.markdown('​当$S_{ij}=0$时有：')
st.latex(r'''\begin{aligned}C=\frac{1}{2}\sigma\left(s_{i}-s_{j}\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)\end{aligned}''')
st.markdown('​当$S_{ij}=-1$时有：')
st.latex(r'''\begin{align}C&=\sigma\left(s_{i}-s_{j}\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)\\&=\log \left(e^{\sigma\left(s_{i}-s_{j}\right)})\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)\\&=\log \left(1+e^{\sigma\left(s_{i}-s_{j}\right)})\right)\end{align}''')
st.markdown('​$\sigma=1$时的损失函数图像如下（自变量为$s_i-s_j$）：')
st.image('assets/image-20250702163213780.png')
st.markdown('​假设$s_i=\mathbf x_i^{\\top}\mathbf w,s_j=\mathbf x_i^{\\top}\mathbf w$，$\mathbf w\in \mathbf R^{h\\times 1}$我们可以看一参数更新公式，以$w_k$（$\mathbf w$的第$k$个分量）为例：')
st.latex(r'''\begin{aligned} \frac{\partial C(s_i,s_j)}{\partial w_k}&= \frac{\partial C}{\partial s_i}\frac{\partial s_i}{\partial w_k}+\frac{\partial C}{\partial s_j}\frac{\partial s_j}{\partial w_k}\\&=\bigg(\frac{1}{2}\left(1-S_{i j}\right) \sigma+\frac{-\sigma e^{-\sigma(s_{i}-s_{j})}}{1+e^{-\sigma\left(s_{i}-s_{j}\right)}}\bigg)\frac{\partial s_i}{\partial w_k}+\bigg(-\frac{1}{2}\left(1-S_{i j}\right)\sigma +\frac{\sigma e^{-\sigma(s_{i}-s_{j})}}{1+e^{-\sigma\left(s_{i}-s_{j}\right)}}\bigg)\frac{\partial s_j}{\partial w_k}\\&=\sigma\bigg(\frac{1}{2}\left(1-S_{i j}\right) -\frac{e^{-\sigma(s_{i}-s_{j})}}{1+e^{-\sigma\left(s_{i}-s_{j}\right)}}\bigg)(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\&=\sigma\bigg(\frac{1}{2}\left(1-S_{i j}\right) -\frac{1}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}\bigg)(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\end{aligned}''')
st.markdown('​且我们可以发现：')
st.latex(r'''\frac{\partial C}{\partial s_{i}}=\sigma\left(\frac{1}{2}\left(1-S_{i j}\right)-\frac{1}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}\right)=-\frac{\partial C}{\partial s_{j}}''')
st.markdown('​因此对应的梯度更新的公式为：')
st.latex(r'''\begin{aligned}w_k\to w_k-\eta\:\frac{\partial C}{\partial w_k}=w_k-\eta\left(\frac{\partial C}{\partial s_i}\frac{\partial s_i}{\partial w_k}+\frac{\partial C}{\partial s_j}\frac{\partial s_j}{\partial w_k}\right)\end{aligned}''')
st.markdown('​损失函数的变化近似为：')
st.latex(r'''\delta C\approx\sum_{k}\frac{\partial C}{\partial w_{k}}\delta w_{k}=\sum_{k}\frac{\partial C}{\partial w_{k}}\left(-\eta\frac{\partial C}{\partial w_{k}}\right)=-\eta\sum_{k}\left(\frac{\partial C}{\partial w_{k}}\right)^{2}<0''')
st.markdown('​即梯度下降一定沿着损失函数减小的方向更新。每次更新都会让损失值降低。然而，初版的$\mathrm{RankNet}$训练效率低下——每次处理一对文档就要更新一次模型，如一个查询有$100$个候选文档，那么两两配对比较就需要$\\begin{pmatrix} 100 \\ 2\\end{pmatrix}$个文档对，这样的计算开销过大。我们回顾上述公式$xxxx$，可以将左边一部分复杂的公式定义：')

st.latex(r'''\lambda_{ij}\equiv\sigma\bigg(\frac{1}{2}\left(1-S_{i j}\right) -\frac{1}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}\bigg)''')
st.markdown('''> :blue[[!NOTE]]
$\lambda_{ij}$代表$i$的关系一定比$j$在前，所以有$S_{ij}=1$，故：
>
>
>
> $$\\begin{aligned}\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\lambda_{ij}&=-\\frac{\sigma}{1+e^{\sigma\\left(s_{i}-s_{j}\\right)}}\\\\ \lambda_{ji}&=-\\frac{\sigma}{1+e^{\sigma\\left(s_{j}-s_{i}\\right)}}\\\\ &=-\\frac{\sigma e^{\sigma\\left(s_{i}-s_{j}\\right)}}{1+e^{\sigma\\left(s_{i}-s_{j}\\right)}}\\\\&=-\sigma(1-\lambda_{ij})\\end{aligned}$$''')
st.markdown('​这样损失函数对单个参数分量的梯度公式就变得清爽了：')
st.latex(r'''\begin{aligned} \frac{\partial C(s_i,s_j)}{\partial w_k}&= \lambda_{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\end{aligned}''')
st.markdown('​我们可以把$\lambda_{ij}$想象成一个作用力，如果模型把本该靠前的文档$doc_i$排在了$doc_j$后面，那么$\lambda_{ij}$就会产生一个力，将$s_i$和$s_j$推开。那么这个作用力是否可以叠加与抵消呢？如果我们找到所有的$\lambda_{ij}$预先计算好这些作用力，那么就可以实现从“逐渐更新”到“批量累计更新”。考虑一个查询下所有的文档对，看看每个权重受到了 多大的推力，并将$w_k$的梯度贡献加起来，有：')
st.latex(r'''\delta w_k=-\eta\sum_{\{i,j\}\in I}\lambda_{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})''')
st.markdown('​现在，这个公式可以写成更加统一的形式：')
st.latex(r'''\begin{aligned}\delta w_k&=-\eta \sum_i\lambda_i(\frac{\partial s_i}{\partial w_k})\\\lambda_i&=\sum_{j:\{i,j\}\in I}\lambda_{ij}-\sum_{j:\{j,i\}\in I}\lambda_{ji}\end{aligned}''')
st.markdown('​意思是对于某个文档$doc_i$，先找到相关性不如它的那些文档$doc_j$，此时可以算出一个向上的叠加的推力即$\\begin{aligned}\sum_{j:\{i,j\}\in I}\lambda_{ij}\\end{aligned}$，同时也会有其他相关性比$doc_i$高的文档，此时$doc_i$上会有一个向下的叠加的拉力即$\\begin{aligned}-\sum_{j:\{j,i\}\in I}\lambda_{ji}\\end{aligned}$。更直观一点，给定$5$个文档，假定关系如下：')
st.image('assets/image-20250703105259182.png')
st.markdown('​那么针对每一个文档$doc_i$，需要计算的$\lambda_{ij}$、$\lambda_{ji}$与$\\frac{\partial s_i}{\partial w_k}$如下表：')
df = pd.DataFrame({
    "文档": ["$doc_1$", "$doc_2$", "$doc_3$", "$doc_4$", "$doc_5$"],
    "$\\lambda_{ij}$": [
        "$\\lambda_{12},\\lambda_{13},\\lambda_{14}$",
        "$\\lambda_{23},\\lambda_{24}$",
        "$\\lambda_{34}$",
        "$\\lambda_{45}$",
        "$\\lambda_{51},\\lambda_{52},\\lambda_{53}$"
    ],
    "$\\lambda_{ji}$": [
        "$\\lambda_{51}$",
        "$\\lambda_{52}$",
        "$\\lambda_{53}$",
        "$\\lambda_{14},\\lambda_{24},\\lambda_{34}$",
        "$\\lambda_{45}$"
    ],
    "$\\frac{\\partial s_i}{\\partial w_k}$": [
        "$\\frac{\\partial s_1}{\\partial w_k}$",
        "$\\frac{\\partial s_2}{\\partial w_k}$",
        "$\\frac{\\partial s_3}{\\partial w_k}$",
        "$\\frac{\\partial s_4}{\\partial w_k}$",
        "$\\frac{\\partial s_5}{\\partial w_k}$"
    ]
})
st.table(df)
st.markdown('​因此，对于一个查询所有的文档，算出他们两两之间的$\lambda_{ij}$，根据公式算出累加梯度$\lambda_i$，所有$\lambda_i$计算完后再根据公式进行梯度更新，显著加速训练速度。即原始的训练方式是遍历所有的文档对$\\begin{pmatrix} 100 \\ 2\\end{pmatrix}$算$O(n^2)$次计算（计算开销小），每次遍历就执行一次梯度更新（计算开销大），有$n^2$次廉价计算加$n^2$次昂贵计算。而改进后有先遍历所有文档对算出$\lambda_i$即$O(n^2)$次计算（计算开销小），再执行$n$次梯度更新，为$n$次昂贵计算，因此将计算复杂度降低至了线性，显著降低计算开销，而这个为加速而生的$\lambda$梯度，启发了研究者们：我们是不是可以绕开复杂的损失函数，直接去定义和优化梯度呢？')
st.markdown('​答案是——可以的，那为什么要直接定义梯度？因为$\mathrm{RankNet}$的优化目标只是成对损失函数，而衡量排序好坏的指标如$\operatorname{NDCG},\operatorname {MRR}$并不是简单的成对损失，因此优化成对损失并不能保证训练后的模型在这些衡量指标上的效果就一定更好。那能否直接优化这些指标呢？——答案是可以，不过很麻烦，因为这些指标的计算涉及到排序算子，排序是一个不可导的操作，没法计算损失函数的梯度并反向传播，需要找到一些可导近似函数进行优化。因此$\mathrm{LambdaRank}$提出不显示定义损失函数而是直接定义梯度来训练神经网络，$\mathrm{LambdaRank}$在$\mathrm{RankNet}$的基础上对$\lambda_{ij}$进行了改造，直接定义梯度为：')
st.latex(r'''\lambda_{i j}=\frac{\partial C(s_i-s_j)}{\partial s_i}=-\frac{\sigma}{1+e^{\sigma\left(s_{i}-s_{j}\right)}} \cdot|\Delta \mathrm{NDCG}|''')
st.markdown('​其中，$|\Delta \mathrm{NDCG}|$是交换$i$与$j$排名后$\mathrm{NDCG}$发生的变化，此时我们不再是让损失最小，而是要让向上的推力越大，使得模型预测的$\mathrm{NDCG}$越大越好，因此更新参数时是梯度上升：')
st.latex(r'''\begin{aligned} w_k\leftarrow w_k+\eta\frac{\partial C}{\partial w_k}=w_k+\eta\sum_i\lambda_i\frac{\partial s_i}{\partial w_k}\end{aligned}''')
st.markdown('​我们把$C$看作一个隐式收益，此时$C$的变化量近似为：')
st.latex(r'''\delta C\approx\frac{\partial C}{\partial w_k}\delta w_k=\eta\big(\frac{\partial C}{\partial w_k}\big)^2\gt 0''')
st.markdown('​因此能说明这个隐式的收益说可以不断变大的。$\mathrm{LmabdaRank}$除了可以优化$\mathrm{NDCG}$指标，还能拓展到其他的指标如$\mathrm{MAP},\mathrm{MRR}$等。只要将$|\Delta \mathrm{NDCG}|$替换成对应的$\mathrm{IR}$指标即可。')
st.markdown('# 3.List-Wise')
st.markdown('​listwise方法可以分为两类，位置有关的指标优化与位置无关的指标优化[[3]]()。和位置有关的衡量指标有$MRR$,$MAP$,$NDCG$等，而模型参数关于这些衡量指标不可导，我们通常采用函数近似的方式构造一个可导函数作为优化目标，从而实现模型参数的更新。首先，我们需要明确的是，什么衡量指标/算子是不可导的？')
st.markdown('## 3.1不可导算子的可导近似')
st.markdown('​学高等数学的时候我们知道导数是指函数描述的是函数在某一处的变化率，可导描述的就是指导数在某一处的变化率是否存在，常见的可到操作有：加减乘除、平方、对数、指数、线性变化、切片等。而不可导就是指函数在某些点处的导数不存在，或者不具备可微性，常见的不可导操作有：阶跃函数、$\\arg\max$​、$\max$、指示函数、排序、采样。')
st.markdown('​在深度学习中，训练神经网络时由于策略的选择原因，标准的优化目标可能涉及到不可导算子，因此通常需要找一个可导算子进行近似。常见的可导算子不可导近似如下：')
df_approx = pd.DataFrame({
    "不可导操作": [
        "$\\max$",
        "$\\arg\\max$",
        "$\\operatorname{Indicator\\ function}$:$I(s_i>s_j)$",
        "$\\operatorname{sort}$",
        "$\\operatorname{sampling}$"
    ],
    "可导近似": [
        "$\\operatorname{log\\ sum\\ exp}$",
        "$\\sum_{i=1}^{n}i*\\operatorname{softmax}(\\mathbf x)_i$",
        "$p(s_i>s_j)$",
        "$\\operatorname{Sinkhorn\\ Operator}$",
        "$\\operatorname{Gumbel\\text{-}softmax}$"
    ]
})
st.table(df_approx)
st.markdown('​以表格中的$\max$不可导算子为例，$\operatorname{max}$ 算子的作用是从一个向量中获得最大值，如$\mathbf v=(2,3,4,1,4,5)^{\\top}$的最大值是$5$，则$\max \mathbf v=5$，其近似如下：')
st.latex(r'''\max(x_1,x_2,...,x_n)\approx\lim_{\tau\rightarrow \infin}\frac{1}{\tau}\log\sum_{i=1}^{n}\exp(\tau x_i)''')
st.markdown('​$\tau$越大则近似越好，当$\tau$取$1$时则$\max$算子的近似就是$\operatorname{log sum exp}$。$\operatorname{sort}$算子和采样算子本文将会在接下来的章节详细介绍。')
st.markdown('## 3.2 SoftRank')
st.markdown('​$\\text{SoftRank}$[[3]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf)的思想是过文档的得分和排名进行概率建模，实现了对$\\text{NDCG}$等指标的可微近似，从而使得梯度下降等优化方法得以应用。$\\text{NDCG}$指标计算依赖于$\\text{DCG}$和$\\text{IDCG}$，但是$\\text{DCG}$这个指标中涉及到了$\mathrm {sort}$的操作是不可导算子，因此训练时没法直接反向传播，如果将$\\text{DCG}$和$\\text{IDCG}$有一个平滑点的可导函数近似，那$\\text{NDCG}$自然也就可导了。给定神经网络预测的分数向量$\mathbf s$，相关性标签$\mathbf r$，$\\text{NDCG}@k(\mathbf s,\mathbf r)$计算方式如下：')
st.latex(r'''\begin{aligned} \text{NDCG@}k(\mathbf s,\mathbf r)&=\frac{\text{DCG@}k(\mathbf s,\mathbf r)}{\text{IDCG@}k(\mathbf r)}\\\text{DCG@}k(\mathbf s,\mathbf r)&=\sum_{j=1}^{k}g(r_{\pi^{-1}(j)})D(j)\\\text{IDCG@}k(\mathbf r)&=\sum_{j=1}^{k}g(r_{\pi_\mathbf r^{-1}(j)})D(j)=\max \text{DCG@}k(\mathbf s,\mathbf r) \end{aligned}''')
st.markdown('​其中$g(\cdot)$是增益因子，$g(z)=2^z-1$，$D(j)=1/\log(j+1)$是位置折扣因子。$\pi^{-1}(j)$是指排序$\pi$中第$j$个位置对应的原来文档的索引，$\pi$是神经网络预测的分数向量$\mathbf s$对应的排序后的列表，$\pi_{\mathbf r^{-1}(j)}$是指相关性分数列表$\mathbf r$从大到小排序得到排序$\pi_{\mathbf r}$中第$j$个位置对应的原来文档的索引。$\\text{IDCG@}k(\mathbf r)$就是最大化$\\text{DCG@k}$，可以看到其和模型预测的分数是无关的，因此给定相关性标签$\mathbf r$，$\\text{IDCG@}k(\mathbf r)$可以被预先计算出。')
st.markdown('​举例：假设当前有$5$个文档，对应的相关性标签$\mathbf r=[0,1,2,0,3]$，神经网络预测的分数$\mathbf s=[0.02,0.01,0.41,0.22,0.31]$，那么先计算$\\text{DCG@k}$再计算$\\text{IDCG@k}$。依据神经网络预测得分排序得到$[0.41,0.31,0.22,0.02,0.01]$，对应的$\pi^{-1}=[3,5,4,1,2]$（假定索引从$1$开始），则对应的增益为：')

df_ranking = pd.DataFrame({
    "排序后文档顺序": ["文档3", "文档5", "文档4", "文档1", "文档2"],
    "标签分数": [2, 3, 0, 0, 1],
    "增益": [3, 7, 0, 0, 1],
    "位置折扣因子": ["$\\log_{2}2=1$", "$\\log_{2}3$", "$\\log_{2}4$", "$\\log_{2}5$", "$\\log_{2}6$"],
    "原来的索引$\\pi^{-1}(j)$": [3, 5, 4, 1, 2]
})

st.table(df_ranking)
st.markdown('​因此$\\text{DCG@}5(\mathbf s,\mathbf r)=\\frac{3}{1}+\\frac{7}{\log_23}+\\frac{0}{\log_24}+\\frac{0}{\log_25}+\\frac{1}{\log_26}$。$\\text{IDCG@}k(\mathbf r)=\\frac{7}{1}+\\frac{3}{\log_23}+\\frac{1}{\log_24}+0+0$。$\\text{SoftRank}$将$\\text{DCG@k}$用概率近似得到$$\\text{SoftDCG@k}$$。具体地，假设当前查询$query$有$k$个候选文档集合$\set{doc_j}_{j=1}^{k}$。将$query$与$doc_j$拼接后得到的文本对$\mathbf x_j$送入一个$\mathrm{Encoder}$，得到神经网络输出的$k$个分数$f(\mathbf \\theta,\mathbf x_j),j=1,..,k$，$\\text{SoftRank}$假设当前文本对$\mathbf x_j$的输出分数$s_j$不再是确定的，而是服从于高斯分布：')
st.latex(r'''\begin{aligned} s_j \sim \mathcal N(s_j|f(\mathbf \theta,\mathbf x_j),\sigma_s^2)\end{aligned}''')
st.markdown('​如果给定两个文档$doc_i$与$doc_j$，从各自的高斯分布中采样得到的分数为$S_i,S_j$,我们想判断谁和$query$更加相似，那么就可以判断$S_i$与$S_i$谁大，但是由于分数是一个随机变量，因此我们看的是一个概率$P(S_i>S_j)$，即$\mathrm{Pr}(S_i-S_j)>0$，而服从高斯分布的随机变量之差仍然是高斯分布，我们定义文档$i$打败文档$j$的概率$\pi_{ij}$：')
st.latex(r'''\begin{aligned} \pi_{ij} :=  \operatorname{Pr}(S_i-S_j>0)=\int_{0}^{\infin} \mathcal N(s|\bar {s_i}-\bar{s_j},2\sigma_s^2) \operatorname {d}s \end{aligned}''')
st.image('assets/learning2rank/Gaussian-area.png')
st.markdown('​我们可以基于成对比较的方式来近似排序，其背后直觉如下：如果文档$doc_j$的排名比较靠后，说明其和其他文档在对比时都被打败了，具体地：假设共计$5$个文档，$doc_2$的排名为$4$（从$0$开始排），说明$doc_2$在和其他四个比较时都被打败了，如果其他文档$doc_j$打败$doc_2$的概率较大，则说明$\pi_{i2},i\\neq2$较大，当$\\bar{s_i}-\\bar{s_2}$较大时高斯分布大于$0$部分的面积接近$1$，此时有$r_{doc_2}=4\\approx \sum_{i=1,i\\neq 2}\pi_{i2}$。因此任意文档$j$的排序$r_j$的期望可以表示如下：')
st.latex(r'''\begin{aligned} \mathbb E\big[r_j\big]= \sum_{i=1,i\neq j}^N\pi_{ij}=\sum_{r=0}^{N-1}rP(r_j=r)\end{aligned}''')
st.markdown('​我们可以将上述式子看作一个$N-1$次的相互独立的伯努利实验，$r_j\sim\operatorname{Bernoulli}(\pi_{ij}),j\\neq i$。但整体而言与标准的二项分布有所不同，标准的二项分布的概率质量函数有一个明确的解析式：')
st.latex(r'''P(X=k)=\begin{pmatrix} n \\ k\end{pmatrix}p^k(1-p)^{n-k}''')
st.markdown('​现在，文档$doc_j$的位置可以看成一个随机变量的期望，将$\mathrm{DCG}$指标中的$D(r_j)$用$\mathbb E\\big[D(r_j)\\big]$替代，则我们可以得到一个可导的计算指标$\operatorname{SoftNDCG}$，即：')
st.latex(r'''\begin{aligned}DCG&=\sum_{i=1}^{N}g(j)D(r_j)\\&\approx \sum_{i=1}^{N}g(j)\sum_{r=0}^{N-1}D(r_j)P(r_j=r)\end{aligned}''')
st.markdown('​只要知道$P(r_j=k)$就能计算$\mathrm{DCG}$，文档$j$的排序位置$r_j$的取值可能为$0,...,N-1$，但是我们会发现情况有点复杂，即$P(r_j=k)$的解析式表达起来很繁琐，当$r_j$取值为$0$时虽然有$P(r_j=0)=\prod_{i=1,i\\neq j}^N(1-\pi_{ij})$，但是当$r_j=1$时可能是$N-1$种情况，即:')
st.latex(r'''P(r_j=1)=\sum_{k=1,k\neq j}^{N}\big(\pi_{kj}\prod_{i=1,i\neq j，i\neq k}^{N}(1-\pi_{ij})\big)''')
st.markdown('​当$r_j=2$时，则有$\\begin{pmatrix} N-1 \\ 2\\end{pmatrix}$种情况，$r_j=3$时有$\\begin{pmatrix} N-1 \\ 3\\end{pmatrix}$种情况，更一般的，我们可以将$P(r_j=k)$表达如下：')
st.latex(r'''P(r_j=k)=\sum_{\substack{E\subseteq\{1,2,...,N\} \setminus \{j\}\\|E|=K}}^{}\big(\prod_{e\in E}\pi_{ej}\prod_{i=1,i\neq j，i\neq e}^{N}(1-\pi_{ij})\big)''')
st.markdown('​该概率质量函数虽然是解析式，但计算时需遍历子集，属于“非封闭形式”（因涉及到组合爆炸）。通常我们需要迭代进行求解，现在来思考一下不同视角下的$P(r_j=k)$的表达方式，从一开始，假设只有一个文档$doc_j$，那么第一次排序其排在位置$0$的概率必然是$1$，现在有第二个文档$doc_i,i\\neq j$进来，我们需要确认文档$doc_j$排在$0$还是$1$，这种情况下如果$doc_j$仍然排在$0$的概率是$1-\pi_{ij}$，排在$1$的概率是从$0$位置跌落一名$\pi_{ij}$。假设有第三篇文档进来，则文档$doc_j$的位置排名只可能出现排序不变及往下跌落一位的情况，不可能上升，如果我们将文档$doc_j$的排序位置视作一个状态，则这个状态只与前一个状态有关，且只可能保持不变或者由前一个状态转移到下一个相邻的状态（第$3$个时刻的位置$3$只能转移到第四个时刻的位置$3$或者位置$4$，不可能转移到位置$2$或位置$5$），整体情况如下图所示：')
st.image('assets/learning2rank/position-transition.png')
st.markdown('​用一个类似于状态转移矩阵的方式刻画（图中灰色圆圈代表文档处于该位置的概率为0），将$P^{(i)}_j(r=k)$记作排序$i,i=1,..j-1,j+1..,N$篇文档时，文档$doc_j$排在位置$k$的概率，则我们可以写一个递推公式：')
st.latex(r'''P_j^{(i)}(r=k)=P_j^{(i-1)}(r=k-1)\pi_{ij}+P_j^{(i-1)}(r=k)\big(1-\pi_{ij}\big)''')
st.markdown('​最终计算得到$P_j^{(N)}(r=k):=P(r_j=k)$，针对所有的$j=1,...,N$,我们都会利用上述公式进行迭代计算得到一个位置分布向量$\mathbf P(r_j)=(P_j(0),P_j(1),$,$....,P_j(N-1))^{\\top}$，再基于公式x求得最终的$\mathrm{SoftNDCG}$​​​，最终用如下公式作为损失函数进行反向传播：')
st.latex(r'''\begin{aligned} \mathcal L(f;\mathbf x,\mathbf r)=1-\frac{1}{Z_m}\sum_{j=1}^{m}(2^r_j-1)\sum_{k=1}^mD(k)P_j(r=k)\end{aligned}''')

st.markdown('## 3.3 $\\text{Approximate Rank}$ & $\\text{SmoothRank}$')
st.markdown('​$\\text{Approximate Rank}$[[5]()认为$\\text{NDCG}$指标不连续的根本原因在于排序的位置关于排序的得分是一个不可导的映射，因此将排序位置用排序分数近似是一个非常直接的想法，具体地，$DCG=\sum_{i=1}^{N}g(j)D(r_j)=\sum_{i=1}^{N}g(j)/\log(1+\pi(\mathbf x_j))$，而$\pi(\mathbf x_j)$是文档在按照模型预测的相关性分数排序后的列表中的位置，从分数到位置的这个操作是不可导的，我们可以把$\pi(\mathbf x_j)$用$s_j=f(\\theta,\mathbf x_j)$进行近似：')
st.latex(r'''\begin{aligned}\pi(\mathbf x_j)&=1+\sum_{i=1,i\neq j}^N\operatorname I\set{s_i>s_j}\\&\approx 1+\sum_{i=1,i\neq j}^NP(s_i-s_j>0)\end{aligned}''')
st.markdown('​其中指示函数$\mathbf I$可以用概率近似，$\\text{Approximate Rank}$提出以如下方式近似$\pi(\mathbf x_j)$：')
st.latex(r'''\begin{aligned}\hat \pi(\mathbf x_j)&=1+\sum_{i=1,i\neq j}^NP(s_i-s_j>0)\\&=1+\sum_{i=1,i\neq j}\frac{\exp(-\alpha(s_j-s_i))}{1+\exp(-\alpha(s_j-s_i))}\end{aligned}''')
st.markdown('​故最后的损失函数是：')
st.latex(r'''\begin{aligned}\mathcal L(f;\mathbf x,\mathbf r)&=1-\text{DCG}_{\max}^{-1}\sum_{j=1}^{N}g(j)/\log(1+\hat\pi(\mathbf x_j))\end{aligned}''')
st.markdown('​$\\text{SmoothRank}$[[6]]()的思想与$\\text{Approximate Rank}$类似，都是基于近似$\pi(\mathbf x_j)$的思想，区别在于近似时的概率质量函数不同，$\\text{SmoothRank}$的具体近似公式如下：')
st.latex(r'''\begin{aligned}&\sum_{j=1}^{m}\sum_{u=1}^{m}g(y_{\pi^{-1}(u)})D(u)\mathbf I\{x_j=x_{\pi^{-1}(u)}\}\\&=\sum_{u=1}^{m}g(y_{\pi^{-1}(u)})D(u)\sum_{j=1}^{m}\mathbf I\{x_j=x_{\pi^{-1}(u)}\}\end{aligned}''')
st.markdown('​其中$\mathbf I\{x_j=x_{\pi^{-1}(u)}\}$为指示函数，当文档$x_j$排在位置$u$时才为$1$，否则为$0$。指示函数的近似公式如下：')
st.latex(r'''\begin{aligned}h_{ju}=\frac{e^{-(f(x_j)-f(x_{\pi^{-1}(u)}))^2/\sigma}}{\sum_{k=1}^{m}e^{-(f(x_k)-f(x_{\pi^{-1}(u)}))^2/\sigma}}\end{aligned}''')
st.markdown('​有了$h_{ju}$便能定义平滑版本的$\\text{NDCG}$指标，并定义损失函数如下：')
st.latex(r'''\begin{aligned} \mathcal L(f;\mathbf x;\mathbf r)&=1-\sum_{j=1}^{m}\sum_{u=1}^{m}g(y_{\pi^{-1}(u)})D(u)\mathbf I\{x_j=x_{\pi^{-1}(u)}\}\\&=1-\sum_{u=1}^{m}g(y_{\pi^{-1}(u)})D(u)\sum_{j=1}^{m}\mathbf I\{x_j=x_{\pi^{-1}(u)}\}\\&=1-\sum_{u=1}^{m}g(y_{\pi^{-1}(u)})D(u)\sum_{j=1}\frac{e^{-(f(x_j)-f(x_{\pi^{-1}(u)}))^2/\sigma}}{\sum_{k=1}^{m}e^{-(f(x_k)-f(x_{\pi^{-1}(u)}))^2/\sigma}}\end{aligned}''')
st.markdown('​值得注意的是$\\text{DCG}_{\max}^{-1}$不见了，且与$\\text{Approximate Rank}$不同的是——不直接近似位置带入$D(u)$，而是乘在外面作为$g(y_{\pi^{-1}(u)})D(u)$的系数，可能是因为')
st.image('assets/learning2rank/SmoothRank.png')
st.markdown('​从图像中可以看到，如果$r$比较小时，随便波动带来的误差较大，当$r$比较大时，对误差的波动就相对没那么敏感了。即折扣函数对排名靠前的位置更敏感，而对靠后位置的贡献衰减较慢但变化幅度较小。因此直接近似位置可能导致在文档数量较多时误差累积增大，显著改变整体$\\text{DCG@}k$的值。而$\\text{DCG}_{\max}^{-1}$是一个常数，和神经网路具体预测的值无关，可以预先计算，因此加不加上并不影响优化目标。')
st.markdown('## 3.4 ListNet&ListMLE')
st.markdown('​ListNet[[7]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)将排序视作一个概率分布，用交叉熵损失优化排序网络。具体地，ListNet先介绍了排列概率$\\text{(Permutation Probability)}$，假设$\pi$是一个关于$n$个物品的排列，$\Phi$是一个严格单调增函数，给定一个分数列表$\mathbf s$，则排列$\pi$出现对应的概率定义为：')
st.latex(r'''P_{\mathbf s}(\pi):=\prod_{j=1}^n\frac{\Phi(\mathbf s_{\pi(j)})}{\sum_{k=j}^{n}\Phi(\mathbf s_{\pi(k)})}''')
st.markdown('​其中$\mathbf s_{\pi(j)}$是排序$\pi$中位置$j$的得分，假设有一个文档$\set{1,2,3}$的分数列表$\mathbf s=\set{s_1,s_2,s_3}$，则排序$\pi=\langle2,3,1\rangle$出现的概率：')
st.latex(r'''\begin{aligned}P_{\mathbf s}(\pi)=\frac{\Phi(s_2)}{\Phi(s_2)+\Phi(s_3)+\Phi(s_1)}\frac{\Phi(s_3)}{\Phi(s_3)+\Phi(s_1)}\frac{\Phi(s_1)}{\Phi(s_1)}\end{aligned}''')
st.markdown('​上述公式只是一个定义式，因为客观世界里排列实际出现的概率和分数列表没有关系，如假设有三个水果分别是“桃子、梨子、苹果”，实际的排列情况有$6$种可能性，每一个排列是等概率的。但是假设将这个水果的排列交给一个人来排列，则有可能人会根据自己的喜好将偏爱的水果放在前面，如果让一个人来排列多次这三个水果，很有可能是喜好的水果在前的次数较多，而排列概率模拟了人类排列物品的过程，但人为定义的排列概率是否满足概率分布的要求我们还需要证明，文中给出了引理$2$：')

st.markdown('''> **Lemma 2** *The permutation probabilities $P_{\mathbf s}(\pi)$, $\pi \in \Omega_n$form a probability distribution over the set of permuta*tions, i.e., for each* $\pi \in \Omega_n$， we have $P_{\mathbf s}(\pi)$, and$\sum_{\pi \in \Omega_n}P_{\mathbf s}(\pi)=1$.''')

st.markdown('​给定两个长度皆为$n$分数列表$\mathbf s_1,\mathbf s_2$，则我们可以计算得到两个排列分布向量$\\begin{pmatrix}\pi_{\mathbf s_1}^1\cdots\pi_{\mathbf s_1}^{n!}\\end{pmatrix}^\\top$与$\\begin{pmatrix}\pi_{\mathbf s_2}^1\cdots\pi_{\mathbf s_2}^{n!}\\end{pmatrix}^\\top$。我们可以用一个度量概率分布差异的指标作为损失函数。在实际计算中，由于排列$n$个物品有$n!$种可能性，计算过于复杂，因此我们只考虑物品$j$被排在第一个位置的概率——Top1 Probability。ListNet定义物品$j$被排序在第一个位置的概率公式：')
st.latex(r'''\begin{aligned}P_{\mathbf s}(j)=\sum_{\pi(1)=j,\pi\in\Omega_n}P_{\mathbf s}(\pi)\end{aligned}''')
st.markdown('​我们希望分数越大的物品被排在第一个位置的概率越高，只要计算每一个物品被排在第一个位置的概率$P_{\mathbf s}(k),k=1,...,n$。但即便如此，根据公式计算概率也几乎不可能，因为直剩下的$n-1$个物品排序仍然有$n-1!$种可能性，文中的定理$6$则明确告诉我们可以通过如下式子计算$P_{\mathbf s}(j)$：')
st.latex(r'''P_{\mathbf s}(j)=\frac{\Phi(\mathbf s_j)}{\sum_{k=1}^n\Phi(\mathbf s_k)}''')
st.markdown('​此外，我们仍需确保$P_{\mathbf s}(j)$也是符合概率分布的，文中引理7：')
st.markdown('​通过Top1概率，给定一个真实标签的概率分布$\mathbf P_{\mathbf r}^{(i)}$和模型输出的概率分布$\mathbf P_{\mathbf s}^{(i)}$，我们就可以用一个度量分布的指标作为损失函数，这里笔者沿用论文中的符号，查询$q^{(i)}$对应的候选文档集合为$\mathbf d^{(i)}=\set{d^{(i)}_{1},...,d_{n^{(i)}}^{(i)}}$，查询$q^{(i)}$对应文档集合的人工标记相关性分数向量记作$\mathbf r^{(i)}=(r^{(i)}_{1},...,r^{(i)}_{n^{(i)}})$，模型预测的输出为$\mathbf s^{(i)}=(s^{(i)}_{1},...,s^{(i)}_{n^{(i)}})$，我们看一下ListNet模型的损失函数：')
st.image('assets/image-20250709153714737.png')
st.markdown('​假设在标注阶段的每一个文档的相关性分数都是确切的，查询$q^{(i)}$标签的概率分布记作$\mathbf P_{\mathbf r}^{(i)}=(P_{r^{(i)}}(1),...,P_{r^{(i)}}(n))^{\\top}$，模型输出的概率分布记作$\mathbf P_{\mathbf s}^{(i)}=(P_{s^{(i)}}(1),...,P_{s^{(i)}}(n))^{\\top}$，前者是目标分布，后者是真实分布，我们可以找一个度量分布的函数作为损失函数，KL散度。若采用KL散度作为损失，则$\operatorname{D}_{KL}(\mathbf P_{\mathbf r}^{(i)}||\mathbf P_{\mathbf s}^{(i)})$表达如下：')
st.latex(r'''\begin{aligned}\operatorname{D}_{KL}(\mathbf P_{\mathbf r}^{(i)}||\mathbf P_{\mathbf s}^{(i)})&=\sum_{k=1}^{n^{(i)}} P_{r^{(i)}}\log \frac{P_{r^{(i)}}}{P_{s^{(i)}}}\\&=C-\sum_{k=1}^{n^{(i)}} P_{r^{(i)}}\log {P_{s^{(i)}}}\\&=C+H(\mathbf P_{\mathbf r}^{(i)},\mathbf P_{\mathbf s}^{(i)})\end{aligned}''')
st.markdown('​由于标签是固定的，即$C$一直不变，采用KL散度作为损失函数等价于用交叉熵作为损失函数。所以$\\text{ListNet}$的损失函数就是Cross Entropy Loss，即：')
st.latex(r'''\begin{aligned} \mathcal L_{\mathrm{ListNet}}(f;\mathbf x;\mathbf r)&=-\sum_{k=1}^{K}P_{r_k}\log\sum_{k=1}^{K}\frac{\exp(s_k)}{\sum_{j=k}^{K}\exp(s_j)}\\&=-\sum_{k=1}^{K}\frac{r_k}{\sum_{j=1}^{K}r_j}\log\bigg(\sum_{k=1}^{K}\frac{\exp(s_k)}{\sum_{j=1}^{K}\exp(s_j)}\bigg)\end{aligned}''')
st.markdown('​$\\text{ListMLE}$[[8]](https://www.researchgate.net/publication/221345286_Listwise_approach_to_learning_to_rank_-_Theory_and_algorithm)则采用了一个更加直接的方式，以真实的标签顺序排列作为目标，基于极大似然估计的思想设计损失函数，即让某个序列出现的概率最大，给定模型预测的分数向量$\mathbf s$，相关性标签$\mathbf r$，$\pi$是相关性标签从大到小排序，$\mathbf s_{\pi}$是模型预测分数按照相$\pi$进行排序后的向量，那么损失函数可以写作：')
st.latex(r'''\mathcal L_{\text{ListMLE}}(f;\mathbf x;\mathbf r)=-\mathbb \log\prod_{k=1}^{K}\frac{\exp{s_{\pi(k)}}}{\sum_{j=k}^{K}\exp(s_{\pi(k)})}\tag{3-x}''')
st.latex(r'''-\log P(\hat {\mathbf s})=-\log \frac{\exp (4)}{\exp (4)+\exp (2)+\exp (3)+\exp (1)}\frac{\exp (3)}{\exp (3)+\exp (2)+\exp (1)}\frac{\exp (2)}{\exp (2)+\exp (1)}''')
st.markdown('​然而，$\\text{ListNet}$与$\\text{ListMLE}$这类排序模型的优化目标与位置无关，用IR的衡量指标如$\\text{NDCG}$来衡量排序好坏时有不一致的矛盾，因此有学者通过引入位置因子解决这个问题，如$\\text{P-ListMLE}$[[9]](https://dl.acm.org/doi/10.5555/3020751.3020798)。')
st.markdown('## 3.5 Neural Sort&Neural NDCG')
st.markdown('​在上述的ListWise形式的排序中，由于$NDCG$指标的计算关于神经网络的输出是一个不可导的操作，因此不可直接优化，可以通过函数近似替代的方式或者与位置无关的损失函数来优化网络，那有没有研究是找到一个离散的排序的可导近似呢？——NeuralSort[[10]](https://arxiv.org/abs/1903.08850)就是一种“连续松弛”，是排序操作的可导近似。')
st.markdown('​Neural Sort的目标是通过反向传播的方式优化包含$\operatorname{sort}$算子的优化目标，即如下形式：')
st.latex(r'''\begin{aligned} L(\theta,\mathbf s)=f(P_{\mathbf z};\theta)\\\text{where } \mathbf z=\operatorname{sort}(\mathbf s)\end{aligned}''')
st.markdown('​其中，$\mathbf s\in\mathbb R^{n}$是一个$n$元实值向量，$\mathbf z$是一个由$\mathbf s$排序后的置换向量，$P_{\mathbf z}$是一个置换矩阵。在上文中，笔者罗列了不可导算子的可导近似，其中$\operatorname{sort}$算子的可导近似是$\operatorname{Sinkhorn}$算子，$\operatorname{Sinkhorn}$算子是一种 **将非可导的排序操作（如 permutation matrix）变成可导形式** 的方法，它常用于 **可微排序（differentiable sorting）** 或 **可微分配（differentiable assignment）** 的场景中。它的关键是将 **离散的置换矩阵（permutation matrix）** 近似为 **可导的双随机矩阵（doubly stochastic matrix）**。置换矩阵$P_{\mathbf z}$是一个特殊的方阵，用于对向量或矩阵进行排序，一个$n\\times n$的矩阵$P$称之为置换矩阵，当且仅当：')
st.latex(r'''\sum_{j=1}^nP_{ij}=1\text{	}\forall i,\text{}\sum_{i=1}^nP_{ij}=1\text{ }\forall j''')
st.markdown('​给定一个$n$维的置换向量$\mathbf z=(z_1,z_2\cdots z_n)^{\\top}\in\mathbb R^n$，$z_i\in\set{1,2\cdots n}$且两两不同，对应的置换矩阵$P_{\mathbf z}$满足：')
st.latex(r'''P_{\mathbf z}[i,j]=\left\{\begin{array}{ll}1 & \text { if } j=z_{i}(\text{排序中第 i 大的元素是原始的第 j 个元素}) \\0 & \text { otherwise }\end{array}\right.''')
st.markdown('​假设一个输入向量$\mathbf s=(9,1,5,2)^{\\top}$，且**定义**$\operatorname{sort}:\mathbb R^n\rightarrow\mathcal Z_n$算子是一个将$n$维实值向量输入映射到一个降序排列的置换向量的操作，则$\mathbf s$经过$\operatorname{sort}$作用后对应的置换向量是$\mathbf z=\operatorname{sort}({\mathbf s})=(1,3,4,2)^{\\top}$，即第$1$个元素第$1$大，第$3$个元素第$2$大，第$4$个元素第$3$大，第$2$个元素第$4$大，置换向量对应的置换矩阵$P_{\mathbf z}$：')
st.latex(r'''P_{\mathbf z}=\left[\begin{array}{llll}1 & 0 & 0 & 0 \\0 & 0 & 1 & 0 \\0 & 0 & 0 & 1 \\0 & 1 & 0 & 0\end{array}\right],P_{\mathbf z}\cdot \mathbf s=(9,5,2,1)^{\top}''')


st.markdown('''
> :red[[!IMPORTANT]]
>
> :blue-background[从列的视角看：]
>给定一个输入向量$\mathbf s$，其对应一个置换向量$\mathbf z$和置换矩阵$P$，则置换矩阵$P$的元素$P[i,j]=1$时的含义是输入向量$\mathbf s_j$一定是第$i$大的。如上案例，$P[1,1]=1$，$\mathbf s_1=9$是第$1$大的。$P[2,3]=1$，$\mathbf s_3= 5$是第$2$大的，$P[3,4]=1$，$\mathbf s_4=2$是第$3$大的，$P[4,2]=1$，$\mathbf s_2=1$是第$4$大的。即如果输入$\mathbf s_j$是第$i$大的，则矩阵第$j$列第$i$行为$1$。
>
> :blue-background[从行的视角看：]
>第 $i$ 行告诉你第 $i $大的元素来自哪个原始索引。''')

st.markdown('​给定任意$\mathbf s$我们需要先找其和到$P_{\operatorname{sort}(s)}$明确的数学表达关系，我们知道的是$P_{\operatorname{sort}(s)}[i,j]=1$一定代表排序后第$i$大的元素对应于原始索$j$，第$1$大可以用$\max$，最小可以用$\min$，但是第$i$大这个该如何通过数学公式描述？我们需要借用这样一个引理：')

st.markdown('''> For an input vector $\mathbf s = [s_1, s_2, \cdots,s_n] ^{\\top}$ that is sorted as $s[1] ≥ s[2] ≥\cdots≥s[n]$ , we have the sum of the $k$-largest elements given as:
> 
>$$\\begin{aligned}\quad\quad\quad\quad\quad\quad\sum_{i=1}^{k} \mathbf s_{[i]}=\min _{\lambda \in\\left\{s_{1}, s_{2}, \ldots, s_{n}\\right\}} \lambda k+\sum_{i=1}^{n} \max \\left(\mathbf s_{i}-\lambda, 0\\right)\\end{aligned}$$''')


st.markdown('​这个引理的证明很简单：')
st.latex(r'''\begin{aligned} \sum_{i=1}^{k} \mathbf s_{[i]}&=\sum_{i=1}^{k} \mathbf s_{[i]}-\lambda+\lambda k \\&\leq\lambda k+\sum_{i=1}^{k} \mathbf \max(s_{[i]}-\lambda,0)\\&\leq \lambda k+\sum_{i=1}^{n} \mathbf \max(s_i-\lambda,0)\end{aligned}\tag{3-x}''')

st.markdown('​当$\lambda$比$ s_{[k]}$小时，$\max$算子是生效的，当$\lambda$等于$s_{[k]}$时，$\max$算子不生效，有：')
st.latex(r'''\begin{aligned} \lambda k+\sum_{i=1}^{n} \mathbf \max(s_{i}-\lambda,0)&=\lambda k +\sum_{i=1}^{n} \mathbf s_{i}-\lambda\\&=k\mathbf s_{[k]}+\sum_{i=1}^{k} \mathbf s_{[i]}-s_{[k]}\\&=\sum_{i=1}^{k} \mathbf s_{[i]}\end{aligned} \tag{3-x}''')
st.markdown('​更具体地，等式$(3-x)$成立的条件是$\mathbf s_{[k]}\leq \lambda \leq \mathbf s_{[k+1]}$。通过控制$\lambda$的大小，我们可以得到$\mathbf s$的前$k$个最大值之和。而第$k$大的值$\mathbf s_{[k]}$表明上可以通过下式得到：')
st.latex(r'''\begin{aligned}\mathbf s_{[i]}&=\sum_{i=1}^{k} \mathbf s_{[i]}-\sum_{i=1}^{k-1} \mathbf s_{[i]}\\&=\min _{\lambda \in\left\{s_{1}, s_{2}, \ldots, s_{n}\right\}} \lambda k+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda, 0\right)\\&-\big(\min _{\lambda'\in\left\{s_{1}, s_{2}, \ldots, s_{n}\right\}} \lambda'(k-1)+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda' 0\right) \big)\\&=\min_{\lambda}F_{k}(\lambda)-\min_{\lambda'}F_{k-1}(\lambda')\end{aligned}''')
st.markdown('​我们会发现这是一个差分$\min$运算，会让优化问题变得复杂，不便于接下来的推导。为此，我们必须想一种方式让优化目标只有一个$\min$。等式$\sum_{i=1}^{k} \mathbf s_{[i]}=\lambda k+\sum_{i=1}^{n} \mathbf \max(s_{i}-\lambda,0)$的成立条件是$\mathbf s_{[k]}\leq \lambda \leq \mathbf s_{[k+1]}$，我们思考是否可以再构造一个优化目标使得$\mathbf s_{[k-1]}\leq \lambda \leq \mathbf s_{[k]}，$这样就可以通过夹逼的方式强迫$\lambda=\mathbf s_{[k]}$，优化$\lambda$使得目标最小就得到了最终的$\mathbf s_{[k]}$。换个角度想，前$k$大其实等价于后$n-k+1$小，如果将$\mathbf s$取负即令$\mathbf t=-\mathbf s$，则$\mathbf t$的前$n-k+1$大等价于$\mathbf s$的后$n-k+1$小等价于$\mathbf s$的前$k$​大。如下是一个直观的例子：')
st.image('assets/learning2rank/reverse-index.png')
st.markdown('​因此可以同样使用引理2把$\mathbf t$的前$n-k+1$大的和写成：')
st.latex(r'''\begin{aligned} \sum_{i=1}^{n-k+1}\mathbf t_{[i]}&=\min_{\lambda\in\mathbf t=-\mathbf s}\big[ \lambda(n-k+1)+\sum_{i=1}^{n}\max(\mathbf t_i-\lambda,0)\big]\\&st.\mathbf t_{[n-k+1]}\leq \lambda \leq \mathbf t_{[n-k+2]}\end{aligned}''')
st.markdown('​再令$\lambda=-\lambda$​，则有：')
st.latex(r'''\begin{aligned} \sum_{i=1}^{n-k+1}\mathbf t_{[i]}&=\min_{\lambda\in\mathbf -t=\mathbf s}\big[ -\lambda(n-k+1)+\sum_{i=1}^{n}\max(\mathbf \lambda-s_i,0)\big]\\&st.\mathbf t_{[n-k+1]}\leq -\lambda \leq \mathbf t_{[n-k+2]}\equiv \mathbf s_{[k]}\geq\lambda \geq \mathbf s_{[k-1]}\end{aligned}''')
st.markdown('​但本质上我们是想求得$\lambda$，因此我们合并两个$\\arg\min_{\lambda}$使得$ \mathbf s_{[k]}\geq\lambda \geq \mathbf s_{[k-1]}$与$\mathbf s_{[k]}\leq \lambda \leq \mathbf s_{[k+1]}$同时成立，即$\lambda^*=\mathbf s_{[k]}$，最终合并两式和，有关于$\lambda$的优化目标为：')
st.latex(r'''\begin{aligned} \lambda^*=\mathbf s_{[k]}&=\arg \min _{\lambda \in \mathbf{s}}\left(\lambda k+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda, 0\right)+\lambda(k-1-n)+\sum_{i=1}^{n} \max \left(\lambda-\mathbf s_{i}, 0\right)\right) \\&=\arg \min _{\lambda\in\mathbf s}\lambda(2k-1-n)+\sum_{i=1}^{n}\max(\mathbf s_i-\lambda,0)+\max(\mathbf \lambda-\mathbf s_i,0)\\&=\arg \min _{\lambda\in\mathbf s}\lambda(2k-1-n)+\sum_{i=1}^{n}|\mathbf s_i-\lambda|\\&=\arg \max _{\lambda\in\mathbf s}\lambda(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\lambda|\end{aligned}''')
st.markdown('​我们看这个等式，并把它展开：')
st.latex(r'''\begin{aligned} \lambda^*&=\arg \max _{\lambda\in\mathbf s}\lambda(n+1-2k)-\sum_{i=1}^{n}| s_i-\lambda|\\&=\arg \max _{\lambda\in\mathbf s}\begin{pmatrix}s_1(n+1-2k)-\sum_{i=1}^{n}|s_i-\mathbf s_1| \\s_2(n+1-2k)-\sum_{i=1}^{n}| s_i-\mathbf s_2|\\\vdots \\s_n(n+1-2k)-\sum_{i=1}^{n}| s_i-\mathbf s_n|\end{pmatrix}\\&=\arg \max _{\lambda\in\mathbf s}[(n+1-2k)\mathbf s-\mathbf A_i\mathbb 1]\end{aligned}''')
st.markdown('​该式子的含义是遍历$\lambda \in \mathbf s$使得$\lambda(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\lambda|$达到最大，如果抛去$\\begin{aligned}\\arg\max_{\lambda}\\end{aligned}$的下角标$\lambda$，则：')
st.latex(r'''\begin{aligned} &\arg \max \begin{pmatrix}s_1(n+1-2k)-\sum_{i=1}^{n}| s_i-\mathbf s_1| \\s_2(n+1-2k)-\sum_{i=1}^{n}| s_i-\mathbf s_2|\\\vdots \\s_n(n+1-2k)-\sum_{i=1}^{n}| s_i-\mathbf s_n|\end{pmatrix}\\&=\arg \max [(n+1-2k)\mathbf s-\mathbf A_i\mathbb 1]\end{aligned}''')
st.markdown('​作用是找到该向量中分量值最大元素对应的索引。若结果是索引$i$，则说明$s_i=s_{[k]}$，即排序后第$k$大的元素来自于原始索引$i$，理清了这层关系，我们开始构造置换矩阵$P$。我们按照行进构造，即先找排序后第$1$大的元素对应与原始索引是多少，再以此类。选定第$1$行，从左往右，那么我们就判断$\mathbf s_1=\mathbf s_{[k]}$时的$k$是多少，则$P[1,k]=1$，第$k$行之外的其他行$P[i,1]=0,i\\neq k$。那$s_i$第几大我们要一一判断，从$k=1,2,...,n$，对于第$1$​列而言，只要：')
st.latex(r'''\begin{aligned} \arg \max \begin{pmatrix}s_1(n+1-2k)-\sum_{i=1}^{n}| s_i-s_1| \\s_2(n+1-2k)-\sum_{i=1}^{n}| s_i- s_2|\\\vdots \\s_n(n+1-2k)-\sum_{i=1}^{n}| s_i- s_n|\end{pmatrix}=1，\text{for k in }1,2,...,n\end{aligned}''')
st.markdown('​便能说明$\mathbf s_1=\mathbf s_{[k]}$，因此，对于第$1$行我们可以写成：')
st.latex(r'''\begin{aligned}P[1,j]==\left\{\begin{array}{ll}1 & \text { if } j=\arg \max [(n+1-2)\mathbf s-\mathbf A_{\mathbf s}\mathbb 1] \\0 & \text { otherwise }\end{array}\right.\end{aligned}''')
st.markdown('​同理，第二行我们可以写成：')
st.latex(r'''\begin{aligned}P[2,j]==\left\{\begin{array}{ll}1 & \text { if } j=\arg \max [(n+1-2\times 2)\mathbf s-\mathbf  A_{\mathbf s}\mathbb 1] \\0 & \text { otherwise }\end{array}\right.\end{aligned}''')
st.markdown('​以此类推，第$i$行我们可以写成的形式：')
st.latex(r'''\begin{aligned}P[i,j]==\left\{\begin{array}{ll}1 & \text { if } j=\arg \max [(n+1-2i)\mathbf s-\mathbf A_{\mathbf s}\mathbb 1] \\0 & \text { otherwise }\end{array}\right.\end{aligned}''')
st.markdown('​当然，$\\arg\max$算子是不可导的，$P$也是不可导的，$P$的每一行是一个$\operatorname{one-hot}$向量，我们找一个可导近似来近似这个$\operatorname{one-hot}$向量，最简单的，可以用算子$\operatorname{softmax with temperature}$近似，即：')
st.latex(r'''\lim _{\tau \rightarrow 0^{+}} \widehat{P}_{\operatorname{sort}(\mathbf{s})}[i,:](\tau)=P_{\operatorname{sort}(\mathbf{s})}[i,:] \quad \forall i \in\{1,2, \ldots, n\}''')
st.markdown('​温度系数$\tau$越小则该分布越接$\operatorname{one-hot}$向量，$\tau$越大则越接近平稳分布。接下来我们再思考如何构造优化目标，在很多排序任务中，**目标排序是唯一的**，比如给定一个数字列表$[3,2, 1, 4]$，我们知道升序结果是 $[1, 2,3, 4]$。但在**模型学习排序函数**时，并不是直接输出这个固定结果，而是输出一个分数向量，然后间接地产生排序。一个分数向量可能对应多个潜在的排列结果，如果只用一个排列，那么信号太稀疏了，不够全面，网络在学习时可能会记住某个输入对应的一种排列情况而不是学到通用规律。而列举分数向量$\mathbf s$​所有排列情况显然也不现实，因此我们需要采样，即从所有可能性结果中选出一部分具有代表性的排列，然后评估这些排序的表现，用这些反馈去优化分数向量生成器，从而提高神经网络的泛化能力。')
st.markdown('''> :blue[[!NOTE]]
>
> 不采样就对应了确定性Neural Sort，采样并重参数化就对应了随机Neural Sort。
''')
st.markdown('​直接采样排列$z\sim \operatorname{Plackett-Luce}(\mathbf s)$这个操作也是不可导的，最常见的解决方式之一是重参数化，将离散采样过程转化为**确定性函数+噪声扰动**，使梯度能通过连续变量传递而$\operatorname{Gumbel Softmax}$就是代表性的重参数化方法。重参数化方法是处理如下优化目标的一种方法：')
st.latex(r'''\begin{aligned} L_{\theta}=\mathbb E_{z\sim P_{\theta}(z)}\big[ f(z)\big]\end{aligned}''')
st.markdown('​由于采样操作不可导，因此没有办法写一个精确的$L_{\\theta}$，而$z$从$p_{\\theta}(z)$中采样会失去关于参数$\\theta$​的梯度信息，重参数方法则将采样变化成“固定随机数 + 可导变换”的方式使得反向传播可以用于训练目标涉及到采样操作的神经网络。其具体数学形式如下：设一个随机变量 $z∼p(z∣θ)z$，其中 $\\theta$ 是模型参数，例如均值 $\mu$、标准差 $\sigma$等，目标关于参数的梯度为：')
st.latex(r'''\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]''')
st.markdown('​引入一个可以重参数化的随机变量$\epsilon\sim p(\epsilon)$，使得：')
st.latex(r'''z=g_{\theta}(\epsilon)''')
st.markdown('​其中，$g$是一个确定性的可导分布，$\epsilon$的分布和$\\theta$无关，则有：')
st.latex(r'''\begin{aligned}\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]&=\nabla_{\theta}\mathbb E_{\epsilon\sim p(\epsilon)}\big[ f(g_{\theta}(\epsilon))\big]\\&=\mathbb E_{\epsilon\sim p(\epsilon)}\big[ \nabla_{\theta}f(g_{\theta}(\epsilon))\big]\end{aligned}''')
st.markdown('​以常见的高斯分布为例，假设随机变量$z\sim \mathcal N(z;\mu_{\\theta},\sigma_{\\theta})$，令$z=\sigma_{\\theta}\epsilon+\mu_{\\theta}$，其中$\epsilon \sim \mathcal N(\epsilon;0,1)$，则有：')
st.latex(r'''\begin{aligned}\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]&=\nabla_{\theta}\mathbb E_{\epsilon\sim \mathcal N(\epsilon;0,1)}\big[ f(\sigma_{\theta}\epsilon+\mu_{\theta})\big]\\&=\mathbb E_{\epsilon\sim \mathcal N(\epsilon;0,1)}\big[ \nabla_{\theta}f(\sigma_{\theta}\epsilon+\mu_{\theta})\big]\end{aligned}''')
st.markdown('​在离散情况，将随机变量$z$用$y$代替，以从类别分布中采样为例：')
st.latex(r'''\mathbf P_{\theta}=[P_{\theta1},P_{\theta2},...,P_{\theta k}]''')
st.markdown('​现在$y\sim \operatorname{Categorical(\mathbf P_{\\theta})}$，我们需要找一个确定性的可导分布$y=g_{\\theta}(\epsilon)$使得采样的随机性转移到随机变量$\epsilon$上，而$\operatorname{Gumbel Max}$提供了一种从类别分布中采样的方法[[11]](https://kexue.fm/archives/6705)（本节暂不对此进行深入的原理解析）：')
st.latex(r'''\begin{aligned} \arg\max_i &\big(\log  p_{\theta i} -\log(-\log\epsilon_i)\big)^k \\\epsilon_i &\sim U[0,1]\end{aligned}''')
st.markdown('​即先从均匀分布中先采样得到$k$个随机数，然后再计算每一个$\\big(\log  p_{\\theta i} -\log(-\log\epsilon_i)\\big)^k$，找到最大值对应的索引：')
st.latex(r'''\arg\max_i\begin{pmatrix}\big(\log  P_{\theta 1} -\log(-\log\epsilon_1)\big)^k\\\big(\log P_{\theta 2} -\log(-\log\epsilon_2)\big)^k\\\vdots \\\big(\log  P_{\theta k} -\log(-\log\epsilon_k)\big)^k\end{pmatrix}''')
st.markdown('​前面讲过$\\arg\max$算子不可导，通常用算子$\operatorname{softmax with temperature}$近似，因此得到了$\operatorname{Gumbel Max}$的可导近似$\operatorname{Gumbel Softmax}$，回到主题上，利用$\\text{Gumbel-Max Trick}$，对得分向量$\mathbf s$中每个元素加上独立$\operatorname{Gumbel}$噪声，使得:')
st.latex(r'''\tilde {\mathbf s}=\beta\log\mathbf s_i+g_i,g_i\sim \operatorname{Gumbel}(0,1)''')
st.markdown('​然后对得到的$\tilde {\mathbf s}$进行排序，对应的置换向量为$\tilde {\mathbf z}$，对应的置换矩阵就是$P_{\operatorname{\tilde {\mathbf s}}}$，和确定性排序一样，利用$\operatorname{softmax with temperature}$近似，即：')
st.latex(r'''\lim _{\tau \rightarrow 0^{+}} \widehat{P}_{\operatorname{sort}(\tilde {\mathbf{s}})}[i,:](\tau)=P_{\operatorname{sort}(\tilde {\mathbf{s}}))}[i,:] \quad \forall i \in\{1,2, \ldots, n\}''')
st.markdown('​因此，重参数化后的优化目标可以表示如下：')
st.latex(r'''\begin{aligned}\mathcal L(\theta,\mathbf s)=\mathbb E_{\mathbf g\sim \operatorname{Gumbel}(0,1)}\big[f(\hat P_{\operatorname{sort}(\beta\log \mathbf s+\mathbf g)};\theta)\big]\\\nabla_{\mathbf s}\mathcal L(\theta,\mathbf s)=\mathbb E_{\mathbf g\sim \operatorname{Gumbel}(0,1)}\big[\nabla_{\mathbf s}f(\hat P_{\operatorname{sort}(\beta\log \mathbf s+\mathbf g)};\theta)\big]\end{aligned}''')
st.markdown('​接下来结合$\operatorname{kNN}$的例子加强对该算法的理解，我们先回顾一下原始的$\operatorname{kNN}$，$\operatorname{kNN}$是一种 **非参数（non-parametric）监督学习算法**，既可以用于分类，也可以用于回归。其核心思想非常简单：')
st.markdown('​假设一个分类任务，给定训练样本和对应的标签$\mathcal D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),...,(\mathbf x_n,y_n)\}$，要预测新的样本$\mathbf x_{\\text{new}}$属于哪一个类别，那么只需判断该新样本距离最近的$k$个（$k$是超参数）距离$d$对应的训练数据的类别，再透过投票来决定该样本所述的类别即可。距离$d$可以是欧氏距离、余弦相似度、曼哈顿距离等。具体地，选取一个查询样本$(\mathbf {x_0},y_0)$，随机选取$n$个样本作为其候选邻居${(\mathbf x_1,y_1),\ldots}$,${(\mathbf x_n,y_n)}$，然后神经网络会学到一个映射表示$h_{\phi}(\cdot)$，将输入编码成高维语义向量，即$h_{\phi}(\mathbf x_i)^{\\top}\in\mathbb R^{h\\times 1}$，然后计算查询样本和候选样本在语义空间中的度量：')
st.latex(r'''\mathbf s_j=|| h_{\phi}(\mathbf x_0) -h_{\phi}(\mathbf x_j)||^2_2,j=1,\ldots,n''')
st.markdown('​得到了分数向量便可以采用$\operatorname{NeuralSort}$的松弛排序$\hat P_{\operatorname{sort}(\mathbf s)}$，而策略的选择，即$f$可以用如下公式：')
st.latex(r'''\mathcal l_{\operatorname{kNN}}(\hat P_{\operatorname{sort}(\mathbf s)},y_0,\ldots,y_n)==-\frac{1}{k} \sum_{j=1}^{k} \sum_{i=1}^{n} 1\left(y_{i}=y_{0}\right) \hat{P}_{z}[i, j]''')
st.markdown("本质上就是交叉熵损失。即按照列的方向扫过去，判断前$k$列预测的结果和真实结果是否一致。在推理时，给定测试样本$\mathbf x'_0$，先计算对应的语义表征$\mathbf e_0=h_{\phi}(\mathbf x_0')$,计算训练集中所有点的语义表征$e_j=h_{\phi}(\mathbf x_j),j=1,\ldots,|\mathcal D|$，然后用欧式距离排序，选择$k$​个最近邻居，通过多数投票确定预测标签。原论文还有手写数字识别数据集和分位回归任务的实验，理解了上文的讲解便能触类旁通，笔者就不在此展开介绍了。")
st.markdown('​上文中$2.3$提到$\\text{NDCG}$指标计算依赖于$\\text{DCG}$和$\\text{IDCG}$，$\\text{DCG}$中涉及到了$\mathrm {sort}$的操作是不可导算子，导致神经网络无法优化，那么现在有了$\operatorname{NeuralSort}$便可以自然地想到将可导的置换矩阵用于近似排序算子从而优化$\operatorname{NDCG@k}$，回顾一下公式：')
st.latex(r'''\begin{aligned} \text{NDCG@}k(\mathbf s,\mathbf r)&=\frac{\text{DCG@}k(\mathbf s,\mathbf r)}{\text{IDCG@}k(\mathbf r)}\\\text{DCG@}k(\mathbf s,\mathbf r)&=\sum_{j=1}^{k}g(r_{\pi^{-1}(j)})D(j)\\\text{IDCG@}k(\mathbf r)&=\sum_{j=1}^{k}g(r_{\pi_\mathbf r^{-1}(j)})D(j)=\max \text{DCG@}k(\mathbf s,\mathbf r) \end{aligned}''')
st.markdown('​$\\text{IDCG}@k$就是固定的，可以提前计算好。而$\operatorname{DCG@}k$的计算要依赖于神经网络输出的文档相关性分数向量$\mathbf s$排序，得到排序后的分数向量中对应的$r_i$。给定神经网络的输出分数向量$\mathbf s$，$\mathbf s$对应于一个置换矩阵$P_{\operatorname{sort}(\mathbf s)}$，给$\mathbf s$排序等价于置换矩阵右乘以$\mathbf s$即$P_{\operatorname{sort}(s)}\mathbf s$，为了可导的性质，我们实际上用一个单峰行随机矩阵$\hat P_{\operatorname{sort}(\mathbf s)}(\tau)$来近似真正的$P_{\operatorname{sort}(s)}$（$\operatorname{NeuralSort}$）。')
st.latex(r'''\lim _{\tau \rightarrow 0^{+}} \widehat{P}_{\operatorname{sort}(\mathbf{s})}[i,:](\tau)=\operatorname{softmax}\left[\left((n+1-2 i) \mathbf s-A_{\mathbf s} \mathbb{1}\right) / \tau\right] \quad \forall i \in\{1,2, \ldots, n\}''')
st.markdown('​$\hat P_{\operatorname{sort}(s)}(\tau)$在下文记作。为计算$\operatorname{DCG@}k$，我们要知道按照$\mathbf s$排序后对应的相关性分数标签$r_i$和增益$g_i$，因此有$\hat P g(\mathbf r)$，意思是相关性标签列表按照$\mathbf s$大小进行排序对应的增益。因此最原始情况下的$\operatorname{NeuralNDCG}@k(\tau)(\mathbf {s,r})$公式可以表达如下：')
st.latex(r'''\begin{aligned} \operatorname{NeuralNDCG}@k(\tau)(\mathbf {s,r})=\frac{\sum_{i=1}^{k}\big(\hat P g(\mathbf r)\big)_id(i)}{\operatorname{IDCG@}k}\end{aligned}''')
st.markdown('​$\hat P$矩阵的性质是每一行之和为$1$，每一列之和不一定为$1$，这样的性质会有什么影响呢？我们可以分别看公式：$\\begin{aligned}\sum_{i=1}^{k}P g(\mathbf r)\\end{aligned}$与$\\begin{aligned}\sum_{i=1}^{k}\hat P g(\mathbf r)\\end{aligned}$。前者是文档真实的获得的增益有：')
st.latex(r'''\begin{aligned} \end{aligned}\begin{aligned}\sum_{i=1}^{k}(P g(\mathbf r))_i=\sum_{i=1}^{k}\sum_{j=1}^{k}P[i,j]g(\mathbf r)_j=\sum_{i=1}^{k}g(r_i)\end{aligned}''')
st.markdown('​后者为：')
st.latex(r'''\begin{aligned} \end{aligned}\begin{aligned}\sum_{i=1}^{k}(\hat P g(\mathbf r))_i&=\sum_{i=1}^{k}\sum_{j=1}^{k} \hat P[i,j]g(\mathbf r)_j\\&=\sum_{j=1}^{k}g(r_j)\sum_{i=1}^{k}\hat P[i,j]\\&\neq \sum_{i=1}^{k}g(r_i)\end{aligned}''')
st.markdown('​也就是说近似的置换矩阵每一列之和不为$1$会使得最终计算的增益要么变大要么变小，即某个文档的增益在排名中可能超量也可能少量，导致$\operatorname{NDCG}$指标的计算偏离预期。可以通过$\operatorname {Sinkhorn Scaling}$把$\hat P$进行行列归一化，确保所有文档对排名的贡献为$1$​。具体步骤如下：')
st.markdown('''
**输入：**

- 近似的置换矩阵$\hat P \in \mathbb{R}^{n \\times n}$。
- 最大迭代次数 `max_iter`。
- 收敛阈值 $\epsilon$（如 $10^{-6}$）。

**输出：**
- 双随机矩阵 $S$，即行和与列和均约为1。
**算法步骤：**

1. **初始化：**
    - 设置 $S := \hat P$

2. **迭代归一化：**
  
    - 对于 $k = 1$ 到 `max_iter`：
    
        a. **行归一化：**
        - 对于每一行 $i$：
            - 计算行和 $r_i = \sum_{j=1}^n S_{i,j}$
            - 更新行 $i$：$S_{[i,:]} = S_{[i,:]} / r_i$
        
        b. **列归一化：**
        - 对于每一列 $j$：
            - 计算列和 $c_j = \sum_{i=1}^n S_{i,j}$
            - 更新列 $j$：$S_{[:,j]} = S_{[:,j]} / c_j$
        
        c. **检查收敛：**
        - 计算所有行和 $\mathbf{r} = \sum_j S_{i,j}$，列和 $\mathbf{c} = \sum_i S_{i,j}$。
        - 如果
        $$
        \max\\bigl(\\lvert \mathbf{r} - \mathbf{1} \\rvert \cup \\lvert \mathbf{c} - \mathbf{1} \\rvert \\bigr) < \epsilon
        $$
        则停止迭代。
    
3. **输出：**
  
    - 返回归一化后的置换矩阵 $S$​​。
    
''')
st.markdown('​故改进后的$\operatorname{NeuralNDCG}@k(\tau)(\mathbf {s,r})$公式为：')
st.latex(r'''\begin{aligned} \operatorname{NeuralNDCG}@k(\tau)(\mathbf {s,r})=\frac{\sum_{i=1}^{k}\big(S g(\mathbf r)\big)_id(i)}{\operatorname{IDCG@}k}\end{aligned}''')
st.markdown('# 4.Ranking skills in Direct Preference Optimization')
st.markdown('​预训练——有监督微调——人类反馈强化学习是打造一个高性能大语言模型的标准步骤。在对齐阶段，目前的RLHF技术如PPO在训练时不够稳定，且对计算资源要求高，为此，$\\text{DPO}$技术应运而生，$\\text{DPO}$的思想是将$\\text{RLHF}$中显示的奖励函数转化到统一的有监督损失中，使得模型可以通过有监督的方式微调参数，在给定偏好数据（人类专家对同一输入不同输出的优劣判断）的情况下，直接学习生成更优的输出，从而绕过传统$\mathrm{RLHF}$​中复杂且不稳定的策略优化过程。')
st.markdown('​在实际工作中，大部分的时间都是与收集与清洗数据的工作打交道，高质量的数据对提升模型下游任务性能有最直接的影响，且模型与策略层面的改动相对来说较少。因此，本文将从数据策略的维度介绍DPO，并从Learning to Rank 的视角解析$\\text{DPO}$。站在数据策略的角度，$\\text{DPO}$可以分成数据质量(Data Quality)、偏好反馈(Preference Feedback)、偏好细粒度(Preference Granularity)三个层面[[12]](https://arxiv.org/abs/2503.11701)，本文将先介绍偏好反馈与信息检索领域的技术关联。')
st.markdown('## 4.1 Preference Feedback')
st.markdown('​偏好反馈指的可以分为$\mathrm{PointWise}$反馈、$\mathrm{PairWise}$反馈和$\mathrm{ListWise}$反馈三类。与$\\text{Ranking}$问题一样，$\mathrm{PointWise}$反馈独立评估每个回答的好坏，往往视作一个回归或分类问题，为其打分或标注为正面或负面；$\mathrm{PairWise}$反馈通过构造成对的偏序关系比较两两之间的好坏；而$\mathrm{ListWise}$反馈则考虑了整个文档内的好坏关系，本文着重介绍成对反馈与列表级反馈。')
st.markdown('### 4.1.1 Pair-Wise Feedback')
st.markdown('​成对反馈侧重于比较成对问答的偏序关系，即给定上下文历史$x_i$，和不同的回复$y^{(i)}_j$,$y^{(i)}_k$。判断回答的相对好坏，即$(x_i,y^{(i)}_j)?(x_i,y^{(i)}_k)$。**Rafailov**&**Sharma**[[13]](https://arxiv.org/abs/2305.18290)等人提出的DPO用$\\text{Bradley-Terry}$模型建模偏好的回复$y_1$大于另一个回复的$y_2$概率，即：')
st.latex(r'''\begin{aligned}p^*(y_1\succ y_2)&=\sigma(r(x,y_1)-r(x,y_2))\\&=\frac{1}{1+\exp{\bigg(\beta\log{\frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}}-\beta\log{\frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}}\bigg)}}\end{aligned}''')
st.markdown('​借助极大似然估计的思想$\\begin{aligned}\\arg\max_{\\theta}\log P(X;\\theta)\\end{aligned}$，优化目标$\mathcal L$可以写成如下形式：')
st.latex(r'''\mathcal L_{\mathrm{DPO}}(\pi_{\theta};\pi_{\text{ref}})=-\mathbb E_{(x,y_c,y_r)\sim \mathcal D}\big[\log\sigma\big(\beta\log{\frac{\pi^*(y_c|x)}{\pi_{\text{ref}}(y_c|x)}}-\beta\log{\frac{\pi^*(y_r|x)}{\pi_{\text{ref}}(y_r|x)}}\big)\big]''')
st.markdown('​其实这个损失函数主体形式和$\mathrm{RankNet}$的损失函数一样：')
st.latex(r'''\begin{aligned} \mathcal L_{\mathrm{RankNet}}&=-\mathbb E_{(x,y_i,y_j)\sim \mathcal D}\big[\log \left(1+e^{-\beta\left(s_{i}-s_{j}\right)}\right)\big]\\&=-\mathbb E_{(x,y_i,y_j)\sim \mathcal D}\big[\log \sigma (\beta s_i-\beta s_j)\big]\end{aligned}''')
st.markdown('​其中,$\sigma$是$\mathrm{sigmoid}$函数，可以看到，二者只是变量不同，$\mathrm{RankNet}$中的$s_i$对应了$\mathrm{DPO}$中的$\log{\\frac{\pi^*(y_c|x)}{\pi_{\\text{ref}}(y_c|x)}}$，$s_j$对应$\log{\\frac{\pi^*(y_r|x)}{\pi_{\\text{ref}}(y_r|x)}}$。因此$\mathrm{DPO}$的训练过程可以视作是$\\text{Learning to Rank}$,给定上下文$x_i$，偏好的回复$y_c$和次之的回复$y_r$，DPO是在学习让$y_c$中的$\\text{token}$的概率排到$y_r$更前面的位置。此外，$DPO$的损失函数可以不只是借助$\\text{Bradley Terry}$模型的概率建模，还可以将信息检索领域的$\mathrm{PairWise}$损失集成，如上文中所介绍的$\mathrm{LambdaRank}$。')
st.markdown('​$\mathrm{DPO}$在训练时会出现正负例奖励同时上升或者下降的情况[[14]](https://zhuanlan.zhihu.com/p/1907949654739513685)。是因为其损失函数只需要正例输出的相对概率比负例大（强调二者间的相对关系，强化二者间的差值，而非绝对大小。），比如$\\frac{0.35}{0.12}-\\frac{0.11}{0.12}$和$\\frac{0.29}{0.11}-\\frac{0.10}{0.11}$都符合正例的相对概率比负例大，但实际上正例的奖励降低了，即存在多种正负例取值的可能满足损失函数在减小但是正负例的奖励增大或者减小。笔者在此给出定量的梯度更新分析，记：')
st.latex(r'''\begin{aligned} A=\beta\log{\frac{\pi^*(y_c|x)}{\pi_{\text{ref}}(y_c|x)}},B=\beta\log{\frac{\pi^*(y_r|x)}{\pi_{\text{ref}}(y_r|x)}}\end{aligned}''')
st.markdown('​求损失函数$\mathcal L=-\log(\sigma(A-B))$对参数分量$w_k$的梯度：')
st.latex(r'''\begin{aligned}\frac{\partial \mathcal L}{\partial w_k}&=\frac{\partial \mathcal L}{\partial A}\frac{\partial A}{\partial w_k}+\frac{\partial \mathcal L}{\partial B}\frac{\partial B}{\partial w_k}\\&=\frac{\sigma(A-B)*(1-\sigma(A-B))}{\sigma(A-B)}*\frac{\partial A}{\partial w_k}-\frac{\sigma(A-B)*(1-\sigma(A-B))}{\sigma(A-B)}*\frac{\partial B}{\partial w_k}\\&=(1-\sigma(A-B))*\beta(\frac{\pi_{\text{ref}}(y_c|x)}{\pi^*(y_c|x)}*\frac{\partial\pi^*(y_c|x)}{\partial w_k}-\frac{\pi_{\text{ref}}(y_r|x)}{\pi^*(y_r|x)}*\frac{\partial\pi^*(y_r|x)}{\partial w_k})\end{aligned}''')
st.markdown('​因此从梯度的角度看，rejected reward的梯度是占据主导地位的，由于损失函数的设计使得模型在优化时无法直接提升chosen reward，因此rejected reward若迅速降低，chosen reward存在不提升，但是慢慢降低的情况，此使使得模型输出的chosen reward仍然大于rejected reward，但是chosen reward的降低会使得模型在训练时逐渐变得不再输出人类偏好的token，训练完的模型会胡说八道，因此在训练过程中需要调整超参数或者引入额外损失等手段解决这个问题，如引入有监督阶段的SFT损失函数，提升模型输出chosen token的概率（DeepSpeed-Chat的RLHF阶段在ppo过程中可以选择性添加预训练阶段任务，即一边ppo让模型的收益增大，一边防止模型能力跑偏，因此在DPO时引入SFT的损失也是可行的手段之一）。')
st.markdown('​DPO运用了Bradley Terry模型建模不同偏好回复的胜负概率，当成对标注出现$y_j=y_k$时，$\\text{Bradley Terry}$模型无法准确建模，成对比较出现打平的情况是十分常见的，如在CBT-Bench中可以看到不同模型和参考答案比较时二者打平的情况其实不在少数，那么如何解决成对比较打平的问题?——引入新的比较模型或是借鉴Learing to Rank的策略。前者的方式是将Bradley Terry 模型替换成可以建模平均概率的$\\text{Rao-Kupper }$模型与$\\text{Davidson}$模型[[15]](https://arxiv.org/pdf/2409.17431)。本文仅以$\\text{Rao-Kupper }$模型为例:')
st.latex(r'''\begin{aligned}p(y_i\succ y_j)&=\frac{\lambda_i}{\lambda_i+\mathcal V\lambda_j}=\frac{1}{1+\mathcal V\lambda_j/\lambda_i}\\p(y_i\sim y_j)&=\frac{(\mathcal V^2-1)\lambda_i\lambda_j}{(\lambda_i+\mathcal V\lambda_j)(\lambda_j+\mathcal V\lambda_i)}=\frac{(\mathcal V^2-1)}{(1+\mathcal V\lambda_j/\lambda_i)(1+\mathcal V\lambda_i/\lambda_j)}\end{aligned}''')
st.markdown('​此时有$p(y_i\succ y_j)+p(y_j\succ y_i)+p(y_i\sim y_j)=1$，是合法的概率密度函数，其中$\mathcal V$用于控制模型分配给平均的概率，我们可以看看$p(y_i\succ y_j)+p(y_j\succ y_i)$与$p(y_i \sim y_j)$的关系，当$\lambda_i=\lambda_j$时：')
st.latex(r'''\begin{aligned} p(y_i\succ y_j)+p(y_j\succ y_i)&=\frac{\lambda_i}{\lambda_i+\mathcal V\lambda_j}+\frac{\lambda_j}{\lambda_j+\mathcal V\lambda_i}\\&=\frac{2\lambda_i\lambda_j+\mathcal V(\lambda_i^2+\lambda_j^2)}{(\lambda_i+\mathcal V\lambda_j)(\lambda_j+\mathcal V\lambda_i)}\\&=\frac{2(\mathcal V+1)\lambda^2}{(1+\mathcal V)^2\lambda^2}\\&=\frac{\mathcal V-1}{2}p(y_i\sim y_j)\end{aligned}''')
st.markdown('​这表明参数$\mathcal V$决定了匹配的项目被判定为平局或不平局的概率，取$\mathcal V=3$有平局概率与不平局概率相同，皆为$0.5$。记$\lambda_i$为$\log{\\frac{\pi^*(y_c|x)}{\pi_{\\text{ref}}(y_c|x)}}$,$\lambda_j$为$\log{\\frac{\pi^*(y_r|x)}{\pi_{\\text{ref}}(y_r|x)}}$。那么此时，基于$\\text{Rao-Kupper }$模型的损失函数可以写作：')
st.latex(r'''\begin{aligned}\mathcal L_{\mathrm{DPO}^{\mathrm{RK}}}(\pi_{\theta};\pi_{\text{ref}})&=-\mathbb E_{(x,y_c\succ y_r)\sim \mathcal D}\big[\log\sigma(\lambda_i-\lambda_j-\alpha)\big]-\\ &\mathbb E_{(x,y_c\sim y_r)\sim \mathcal D}\big[\log\sigma(\lambda_j-\lambda_i-\alpha)+\log(\sigma(\lambda_i-\lambda_j-\alpha))-\log(\mathcal V^2-1)\big] \end{aligned}''')
st.markdown('### 4.1.2 List-Wise Feedback')
st.markdown('​列表级反馈将考虑多个候选回答间的整体关系，即多个回答间的相对顺序。上一小节我们发现了DPO实际上可以看作是在排序学习，那么，在List-Wise FeedBack中我们同样可以引入传统的learning to rank策略，如像$\mathrm{ListMLE}$一样直接优化排序出现的似然，即给定模型预测的分数向量$\mathbf s$，按照$\mathbf s$排序得到的置换向量是$\pi$排序后得到$\mathbf s_{\pi}$，$s_{\pi(k)}$是$\mathbf s_{\pi}$的第$k$个分量，也是第$k$大的元素，那么损失函数可以写作：')
st.latex(r'''\begin{aligned} \mathcal L_{\mathrm{ListMLE}}=-\mathbb E_{x,\mathbf y\sim\mathcal D}\big[\log\prod_{k=1}^{K}\frac{\exp{s_{\pi(k)}}}{\sum_{j=k}^{K}\exp(s_{\pi(k)})}\big]\end{aligned}''')
st.markdown('​同样，$\mathrm{ListNet}$损失函数也能用于$\mathrm{DPO}$，给定真实的相关性标签$\mathbf r=(r_1,\ldots,r_K)$，模型预测的分数向量$\mathbf s$，损失函数可以写成：')
st.latex(r'''\begin{aligned} \mathcal L_{\mathrm{ListNet}}&=-\mathbb E_{x,\mathbf y,\psi\sim\mathcal D}\big[\sum_{k=1}^{K}P_{r_k}\log\sum_{k=1}^{K}\frac{\exp(s_k)}{\sum_{j=k}^{K}\exp(s_j)}\big]\\&=-\mathbb E_{x,\mathbf y,\psi\sim\mathcal D}\big[\sum_{k=1}^{K}\frac{r_k}{\sum_{j=1}^{K}r_j}\log\bigg(\sum_{k=1}^{K}\frac{\exp(s_k)}{\sum_{j=1}^{K}\exp(s_j)}\bigg)\big]\end{aligned}''')
st.markdown('​$\\text{LiPO-}\lambda$[[16]](https://arxiv.org/pdf/2402.01878)直接定义$\lambda$梯度，得到优化目标如下：')
st.latex(r'''\mathbb{E}_{x, \mathbf{y}, \psi \sim \mathcal{D}}\left[\sum_{\psi_{i}>\psi_{j}} \Delta_{i, j} \log \left(1+e^{-\left(s_{i}-s_{j}\right)}\right)\right],''')
st.markdown('​其中$\Delta_{ij}$公式如下：')
st.latex(r'''\begin{aligned}\Delta_{i, j}=\left|G_{i}-G_{j}\right| \cdot\left|\frac{1}{D(\pi(i))}-\frac{1}{D(\pi(j))}\right|\end{aligned}''')
st.markdown('​$G_i$是文档的增益，$D(\pi(i))=\log(1+\pi(i))$是折扣因子，$\pi(i)$是按照分数$\mathbf s$排序后文档$y_i$的位置。至此，不难发现$\\text{DPO}$可以无缝融入各种各样的排序损失 ，上文第三章中提到的$\mathrm{NeuralNDCG}$同样也可以用于$\\text{DPO}$，zhao等人提出的$\mathrm{OPO}$[[17]](https://arxiv.org/pdf/2410.04346)便是基于这个思想，借用公式(3-x)，$\mathrm{OPO}$的优化目标可以写成：')
st.latex(r'''\begin{aligned} \begin{aligned} \operatorname{NeuralNDCG}@k(\tau)(\mathbf {s,r})=\mathbb E_{x,\mathbf y,\mathbf r\sim \mathcal D}\bigg[\frac{\sum_{i=1}^{k}\big(S g(\mathbf r)\big)_id(i)}{\operatorname{IDCG@}k}\bigg]\end{aligned}\end{aligned}''')
st.markdown('​$\mathrm{OPO}$列举了不同反馈标注形式下的工作、类型及优化目标，如下图：')
st.image('assets/image-20250709143803491.png')
st.markdown('​此外，$\mathrm{OPO}$基于$\\text{UltraFeedback}$和$\\text{SimPO}$构建了一个有序奖励的数据集，并通过实验结果表明使用多样化的负样本比仅使用最低质量的回答作为负样本更能够提升模型性能。')
st.markdown('# 5.参考文献')
st.markdown('''
[[1]Burges,Shaked,Renshaw.Learning to Rank using Gradient Descent[C]//Proceedings of the 22 nd International Conference on Machine Learning, Bonn, Germany, 2005.](https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf)

[[2]C.Burges.From RankNet to LambdaRank to LambdaMart:An Overview[EB/OL].Microsoft Research Technical Report MSR-TR-2010-82.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)

[[3]Liu.Learning to Rank for Information Retrieval.Springer, Chapter 4 (The Listwise Approach), p.71.](https://link.springer.com/book/10.1007/978-3-642-14267-3)

[[4]Taylor,Guiver,Robertson,et al.SoftRank: Optimising Non-Smooth Rank Metrics[C]//SIGIR LETOR Workshop ’07,Amsterdam, Netherlands.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf)

[[5]Qin,Liu,Li.A General Approximation Framework for Direct Optimization of Information Retrieval Measures[J].Information Retrieval Journal,2009,3(4):375-397.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-164.pdf)

[[6]Chapelle,Wu.Gradient descent optimization of smoothed information retrieval metrics[J]. Information Retrieval Journal. Special Issue on Learning to Rank,2010,13(3):216-235.](https://link.springer.com/article/10.1007/s10791-009-9110-3)

[[7]Cao,Liu,Tsai,et al.Learning to Rank:From Pairwise Approach to Listwise Approach[C]//Proceedings of the 24th International Conference on Machine learning,Corvallis,2007:129-136.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)

[[8]Xia,Liu,Wang,et al.Listwise approach to learning to rank - Theory and algorithm[C]//Proceedings of the 25th International Conference on Mahcine Learning,Helsinki,2008:1192-1199.](https://www.researchgate.net/publication/221345286_Listwise_approach_to_learning_to_rank_-_Theory_and_algorithm)

[[9]Lan,Zhu,Guo,et al.Position-Aware ListMLE: A Sequential Learning Process for Ranking[C]//Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence,Sydney,2017:449-458.](https://dl.acm.org/doi/10.5555/3020751.3020798)

[[10]Grover,Wang,Zweig,et al.Stochastic Optimization of Sorting Networks via Continuous Relexations[J].International Conference on Learning Representations,2019.](https://arxiv.org/abs/1903.08850)

[[11]苏建林.漫谈重参数：从正态分布到Gumbel Softmax[EB/OL].(2019-06-10)-[2025-06-14]](https://kexue.fm/archives/6705)

[[12]Liu,Fang,Hu.A Survey of Direct Preference Optimization.arXiv preprint arXiv:2410.15595v2.](https://arxiv.org/abs/2503.11701)

[[13]Rafailov,Sharma,Mitchell.Direct Preference Optimization: Your Language Model is Secretely a Reward Model.arXiv preprint:2305.18290.](https://arxiv.org/abs/2305.18290)

[[14]akaihaoshuai.基于Qwen3的DPO/KTO/ORPO/Simpo经验总结[EB/OL].(2025-06-09)-[2025-07-07].](https://zhuanlan.zhihu.com/p/1907949654739513685)

[[15]Chen,Yang,Lin,et al.ON EXTANDING DIRECT PREFERENCE OPTIMIZATION TO ACCOMNODATE TIES.arXiv preprint:2409.17431.](https://arxiv.org/pdf/2409.17431)

[[16]Liu,Qin,Wu,et al.LiPO: Listwise Preference Optimization through Learning-to-Rank.arXiv preprint:2402.01878.](https://arxiv.org/pdf/2402.01878)

[[17]Zhao,Wang,Yin,et al.Ordinal Preference Optimization: Aligning Human Preferences via NDCG.arXiv preprint:2410.04346,](https://arxiv.org/pdf/2410.04346)
            ''')