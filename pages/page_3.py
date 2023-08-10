import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.markdown("# 多元高斯分布 ❄️")
st.sidebar.markdown("# 多元高斯分布 ❄️")
st.write("Welcome to my code space~")
st.markdown("&emsp;&emsp;在机器学习领域，多元高斯分布是一种重要而强大的概率分布模型，被广泛应用于数据建模和统计推断。它提供了一种灵活且强大的方法来描述和理解不同变量之间的关系，同时也能够通过概率密度函数对数据进行建模。\
多元高斯分布是一种多维扩展的高斯分布，它可以处理多个特征之间的相关性。与一元高斯分布相比，多元高斯分布能够更好地捕捉到复杂的数据结构，并提供更准确的预测。")

st.markdown("#### 2.1多元高斯分布的定义")
st.markdown("&emsp;&emsp;对于$D$维向量$\mathbf x$,对应的概率密度函数公式如下所示：")
st.latex(r'''\begin{aligned} \mathcal N(\mathbf x|u,\mathbf\Sigma)&=\frac{1}{(2\pi)^D}\frac{1}{|\mathbf\Sigma|^{1/2}}\mathbf {exp}\left\{-\frac{1}{2}(\mathbf x-\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mu) \right\} \\
    &\mathbf\mu \in R^{D \times 1}(\text {均值向量})\\
    &\mathbf\Sigma \in R^{D \times D}(\text{协方差矩阵})\end{aligned}''')
st.markdown("&emsp;&emsp;注意到，花括号中的$\Delta^2=(\mathbf x-\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mu)   $  又称作马氏距离，当$\mathbf\Sigma$ 退化为单位矩阵时，\
    马氏距离退化为欧氏距离。$prml$中额外提到，$\mathbf\Sigma$可以当作对称矩阵看待，因为任何非对称元素在指数运算中都会被消除。简而言之，**非对称运算可以等价为对称运算**。具体的推导过程如下：")
st.latex(r'''\begin{aligned}\text{记}\mathbf\Lambda=\mathbf\Sigma^{-1},\text{那么有}\Lambda_{ij}=\frac{2\Lambda_{ij}}{2}=\frac{\Lambda_{ij}+\Lambda_{ji}-\Lambda_{ji}+\Lambda_{ij}}{2}=\underbrace{\frac{\Lambda_{ij}+\Lambda_{ji}}{2}}_{\Lambda_{ij}^A}+\underbrace{\frac{\Lambda_{ij}-\Lambda_{ji}}{2}}_{\Lambda_{ij}^S}\end{aligned}''')
st.latex(r'''\text{接着有} \begin{cases} \Lambda_{ij}^S=\Lambda_{ji}^S \\
\Lambda_{ij}^A=-\Lambda_{ji}^A\end{cases}\Longrightarrow \Lambda_{ij}^{A}=\frac{\Lambda_{ij}^{A}}{2}-\frac{\Lambda_{ji}^{A}}{2}''')
st.latex(r'''\begin{aligned}\Delta^2&=(\mathbf x-\mu)^T\Sigma^{-1}(\mathbf x-\mu) \\
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)(\Lambda_{ij}^{A}+\Lambda_{ji}^{S})(x_j-\mu_j)  \\ &=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)+\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{S}(x_j-\mu_j)  \\
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)+\underbrace{\frac{1}{2}\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}(x_j-\mu_j)-\frac{1}{2}\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ji}(x_j-\mu_j)}_{两项相抵消} \\ 
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)\\
\end{aligned}''')
st.markdown("&emsp;&emsp;可以看到，最终只留下一项，这一项可以改写为**对称矩阵**$(\mathbf x-\mu)^T\Lambda^{A}(\mathbf x-\mu)$,这也就证明了为什么指数运算内的非对称矩阵的计算可以等价为对称矩阵的运算。")

st.markdown("#### 2.1多元条件高斯分布")
st.markdown("本小节是已知联合概率密度，给定$\mathbf {x_b}$的条件下求$\mathbf {x_a}$的条件概率密度")
st.markdown("&emsp;&emsp;多元高斯分布的一个重要特性是，如果两个变量集联合概率密度服从多元高斯分布，那么给定其中一个集合的情况下另一个集合变量的条件分布也服从多元高斯分布。相似的，边缘概率密度也服从高斯分布。我们将$D$维向量$\mathbf x$分成两部分$\mathbf {x_a} \in R^{M\times 1}$和$\mathbf x_b \in R^{(D-M)\times 1}$，即：")
st.latex(r''' \mathbf x=\begin{pmatrix} \mathbf x_a  \\\mathbf x_b \end{pmatrix},\text{同理有} \mathbf \mu=\begin{pmatrix} \mathbf \mu_a  \\\mathbf \mu_b \end{pmatrix},\mathbf \Sigma=\left( \begin{array} { c c c c } { \mathbf \Sigma_{aa} } & { \mathbf \Sigma_{ab} }  \\ { \mathbf \Sigma_{ba}} & {\mathbf \Sigma_{bb}}   \end{array} \right)''')
st.markdown("&emsp;&emsp;我们一般将协方差矩阵的逆称作精度矩阵，表示如下：")
st.latex(r'''\begin{aligned}\mathbf \Lambda \equiv \mathbf \Sigma^{-1},\mathbf \Lambda=\left( \begin{array} { c c c c } { \mathbf \Lambda_{aa} } & { \mathbf \Lambda_{ab} }  \\ { \mathbf \Lambda_{ba}} & {\mathbf \Lambda_{bb}}   \end{array} \right)\end{aligned}''')
st.markdown("&emsp;&emsp; 思路：我们知道，高斯分布是一个参数模型，如果知道了其均值向量$\mathbf \mu$和协方差矩阵$\mathbf \Sigma$，**那么这个多元高斯分布可以被这两个参数刻画**，所以接下来我们就是为了在给定其中一部分高斯分量的情况下，去求另一部分的均值和协方差。只要能把均值向量和协方差矩阵的表达式明确，那么多元高斯分布的条件概率就能被唯一确认！比如给定$\mathbf x_b$求$p(\mathbf x_a|\mathbf x_b)$,那么我们只需要知道$\mu_{a|b}$和$\mathbf \Sigma_{a|b}$就能确定条件分布。")
st.markdown("&emsp;&emsp;首先，展开概率密度函数对应的指数内的表达式：")
st.latex(r'''\begin{aligned}&-\frac{1}{2}(\mathbf x-\mu)^T\Sigma^{-1}(\mathbf x-\mu)=-\frac{1}{2}\mathbf x^T\Sigma^{-1}\mathbf x -\mathbf x\Sigma^{-1}\mu+C,\text{其中}C\text{是和x无关的} \\
&-\frac{1}{2}(\mathbf x_a-\mu_a)^T\Lambda_{aa}(\mathbf x_a-\mu_a)-\frac{1}{2}(\mathbf x_a-\mu_a)^T\Lambda_{ab}(\mathbf xb-\mu_b) \\
&-\frac{1}{2}(\mathbf x_b-\mu_b)^T\Lambda_{ba}(\mathbf x_a-\mu_a)-\frac{1}{2}(\mathbf x_b-\mu_b)^T\Lambda_{bb}(\mathbf x_b-\mu_b)\end{aligned}\tag{2.1}''')
st.markdown("&emsp;&emsp;展开以后有：")
st.latex(r'''\begin{aligned}&-\frac{1}{2}\mathbf x_{a}^T\Lambda_{aa}\mathbf x_{a} +\mathbf x_{a}^T\Lambda_{aa}\mu_a-\mathbf x_{a}^T\Lambda_{ab}(\mathbf x_b-\mu_b)+C \\ 
&= -\frac{1}{2}\mathbf x_{a}^T\Lambda_{aa}\mathbf x_{a} +\mathbf x_{a}^T\left\{\Lambda_{aa}\mu_a-\Lambda_{ab}(\mathbf x_b-\mu_b)\right\}+C \\ 
&\sim -\frac{1}{2}\mathbf x^T\mathbf \Sigma^{-1}_{a|b}\mathbf x +\mathbf x^T\mathbf\Sigma^{-1}_{a|b}\mu+C \end{aligned}\tag{2.2}''')
st.markdown("&emsp;&emsp;公式$(2.2)$倒数第一二行的两个公式互相比较，发现元素可以一一对应上，因此得出结论：")
st.latex(r'''\begin{aligned} &\Lambda_{aa}=\Sigma^{-1}_{a|b},~\Sigma^{-1}_{a|b}\mu_{a|b}=\Lambda_{aa}\mu_a+\Lambda_{ab}(\mathbf x_b-\mu_b) \\
&\rightarrow \mathbf \Sigma_{a|b}=\Lambda_{aa}^{-1}\\
&\mu_{a|b}=\Sigma^{-1}_{a|b}\left\{\Lambda_{aa}\mu_a+\Lambda_{ab}(\mathbf x_b-\mu_b)\right\}=\mu_a-\Lambda_{aa}^{-1}\Lambda_{ab}(\mathbf x_b-\mu_b) \\
\end{aligned}''')
st.markdown("&emsp;&emsp;为了更好地推导$\Sigma_{a|b}$,我们先引出分块矩阵的逆：")
st.latex(r'''\begin{aligned}\left( \begin{array} { c c c c } { \mathbf A } & { \mathbf B }  \\ { \mathbf C} & {\mathbf D}  \end{array} \right)^{-1}&=\left( \begin{array} { c c c c } { \mathbf M } & { \mathbf -MBD^{-1} }  \\ { \mathbf -D^{-1}CM} & {\mathbf D^{-1}+D^{-1}CMB D^{-1}}  \end{array} \right) \\
\mathbf M~&=~(\mathbf A-\mathbf {BD^{-1}C})^{-1}\end{aligned}''')

mu1 = np.array([0, 0])
sigma1 = np.array([[1, 0.5], [0.5, 1]])

mu2 = np.array([2, 2])
sigma2 = np.array([[1, -0.5], [-0.5, 1]])

weight1 = 0.6
weight2 = 0.4

x = np.linspace(-4, 6, 100)
y = np.linspace(-4, 6, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        vec = np.array([X[i, j], Y[i, j]])
        gauss1 = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma1)))) * np.exp(-0.5 * np.dot(np.dot((vec - mu1), np.linalg.inv(sigma1)), (vec - mu1).T))
        gauss2 = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma2)))) * np.exp(-0.5 * np.dot(np.dot((vec - mu2), np.linalg.inv(sigma2)), (vec - mu2).T))
        Z[i, j] = weight1 * gauss1 + weight2 * gauss2

fig = go.Figure(data=[
    go.Surface(x=x, y=y, z=Z, colorscale='Viridis')
])
st.plotly_chart(fig, use_container_width=True)