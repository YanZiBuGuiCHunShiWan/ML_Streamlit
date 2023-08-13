import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


st.markdown("# 多元高斯分布 ❄️")
st.sidebar.markdown("# 多元高斯分布 ❄️")
st.sidebar.markdown("## 2.1 多元高斯分布的定义")
st.sidebar.markdown("## 2.2 多元条件高斯分布")
st.sidebar.markdown("### 2.2.1 趣味环节")
st.sidebar.markdown("## 2.3 边缘高斯分布")
st.sidebar.markdown("## 2.4 高斯变量的贝叶斯理论")
st.write("Welcome to my code space~")
st.markdown("&emsp;&emsp;在机器学习领域，多元高斯分布是一种重要而强大的概率分布模型，被广泛应用于数据建模和统计推断。它提供了一种灵活且强大的方法来描述和理解不同变量之间的关系，同时也能够通过概率密度函数对数据进行建模。\
多元高斯分布是一种多维扩展的高斯分布，它可以处理多个特征之间的相关性。与一元高斯分布相比，多元高斯分布能够更好地捕捉到复杂的数据结构，并提供更准确的预测。")

st.markdown("#### :blue[2.1多元高斯分布的定义]")
st.markdown("&emsp;&emsp;对于$D$维向量$\mathbf x$,对应的概率密度函数公式如下所示：")
st.latex(r'''\begin{aligned} \mathcal N(\mathbf x|u,\mathbf\Sigma)&=\frac{1}{(2\pi)^D}\frac{1}{|\mathbf\Sigma|^{1/2}}\mathbf {exp}\left\{-\frac{1}{2}(\mathbf x-\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mu) \right\} \\
    &\mathbf\mu \in R^{D \times 1}(\text {均值向量})\\
    &\mathbf\Sigma \in R^{D \times D}(\text{协方差矩阵})\end{aligned}''')
st.markdown("&emsp;&emsp;注意到，花括号中的$\Delta^2=(\mathbf x-\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mu)   $  又称作马氏距离，当$\mathbf\Sigma$ 退化为单位矩阵时，\
    马氏距离退化为欧氏距离。$prml$中额外提到，$\mathbf\Sigma$可以当作对称矩阵看待，因为任何非对称元素在指数运算中都会被消除。简而言之，**非对称运算可以等价为对称运算**。具体的推导过程如下：")
st.latex(r'''\begin{aligned}\text{记}\mathbf\Lambda&=\mathbf\Sigma^{-1},\text{那么有}\\
    \Lambda_{ij}&=\frac{2\Lambda_{ij}}{2}=\frac{\Lambda_{ij}+\Lambda_{ji}-\Lambda_{ji}+\Lambda_{ij}}{2}\\
    &=\underbrace{\frac{\Lambda_{ij}+\Lambda_{ji}}{2}}_{\Lambda_{ij}^A}+\underbrace{\frac{\Lambda_{ij}-\Lambda_{ji}}{2}}_{\Lambda_{ij}^S}\end{aligned}''')
st.latex(r'''\text{接着有} \begin{cases} \Lambda_{ij}^S=\Lambda_{ji}^S \\
\Lambda_{ij}^A=-\Lambda_{ji}^A\end{cases}\Longrightarrow \Lambda_{ij}^{A}=\frac{\Lambda_{ij}^{A}}{2}-\frac{\Lambda_{ji}^{A}}{2}''')
st.latex(r'''\begin{aligned}\Delta^2&=(\mathbf x-\mu)^T\Sigma^{-1}(\mathbf x-\mu) \\
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)(\Lambda_{ij}^{A}+\Lambda_{ji}^{S})(x_j-\mu_j)  \\ &=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)+\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{S}(x_j-\mu_j)  \\
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)\\
&+\underbrace{\frac{1}{2}\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}(x_j-\mu_j)-\frac{1}{2}\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ji}(x_j-\mu_j)}_{两项相抵消} \\ 
&=\Sigma_{i=1}^{D}\Sigma_{j=1}^{D}(x_i-\mu_i)\Lambda_{ij}^{A}(x_j-\mu_j)\\
\end{aligned}''')
st.markdown("&emsp;&emsp;可以看到，最终只留下一项，这一项可以改写为**对称矩阵**$(\mathbf x-\mu)^T\mathbf \Lambda^{A}(\mathbf x-\mu)$,这也就证明了为什么指数运算内的非对称矩阵的计算可以等价为对称矩阵的运算。")

st.markdown("#### :blue[2.2多元条件高斯分布]")
st.markdown("本小节是已知联合概率密度，给定$\mathbf {x_b}$的条件下求$\mathbf {x_a}$的条件概率密度")
st.markdown("&emsp;&emsp;多元高斯分布的一个重要特性是，如果两个变量集联合概率密度服从多元高斯分布，那么给定其中一个集合的情况下另一个集合变量的条件分布也服从多元高斯分布。相似的，边缘概率密度也服从高斯分布。我们将$D$维向量$\mathbf x$分成两部分$\mathbf {x_a} \in R^{M\\times 1}$和$\mathbf x_b \in R^{(D-M)\\times 1}$，即：")
st.latex(r''' \mathbf x=\begin{pmatrix} \mathbf x_a  \\\mathbf x_b \end{pmatrix},\text{同理有} \mathbf \mu=\begin{pmatrix} \mathbf \mu_a  \\\mathbf \mu_b \end{pmatrix},\mathbf \Sigma=\left( \begin{array} { c c c c } { \mathbf \Sigma_{aa} } & { \mathbf \Sigma_{ab} }  \\ { \mathbf \Sigma_{ba}} & {\mathbf \Sigma_{bb}}   \end{array} \right)''')
st.markdown("&emsp;&emsp;我们一般将协方差矩阵的逆称作精度矩阵，表示如下：")
st.latex(r'''\begin{aligned}\mathbf \Lambda \equiv \mathbf \Sigma^{-1},\mathbf \Lambda=\left( \begin{array} { c c c c } { \mathbf \Lambda_{aa} } & { \mathbf \Lambda_{ab} }  \\ { \mathbf \Lambda_{ba}} & {\mathbf \Lambda_{bb}}   \end{array} \right)\end{aligned}''')
st.markdown("&emsp;&emsp; 思路：我们知道，高斯分布是一个参数模型，如果知道了其均值向量$\mathbf \mu$和协方差矩阵$\mathbf \Sigma$，**那么这个多元高斯分布可以被这两个参数刻画**，所以接下来我们就是为了在给定其中一部分高斯分量的情况下，去求另一部分的均值和协方差。只要能把均值向量和协方差矩阵的表达式明确，那么多元高斯分布的条件概率就能被唯一确认！比如给定$\mathbf x_b$求$p(\mathbf x_a|\mathbf x_b)$,那么我们只需要知道$\mu_{a|b}$和$\mathbf \Sigma_{a|b}$就能确定条件分布。")
st.markdown("&emsp;&emsp;首先，展开概率密度函数对应的指数内的表达式：")
st.latex(r'''\begin{aligned}&-\frac{1}{2}(\mathbf x-\mu)^T\Sigma^{-1}(\mathbf x-\mu)=-\frac{1}{2}\mathbf x^T\Sigma^{-1}\mathbf x -\mathbf x^T\Sigma^{-1}\mu+C,\text{其中}C\text{是和x无关的} \\
&-\frac{1}{2}(\mathbf x_a-\mu_a)^T\Lambda_{aa}(\mathbf x_a-\mu_a)-\frac{1}{2}(\mathbf x_a-\mu_a)^T\Lambda_{ab}(\mathbf xb-\mu_b) \\
&-\frac{1}{2}(\mathbf x_b-\mu_b)^T\Lambda_{ba}(\mathbf x_a-\mu_a)-\frac{1}{2}(\mathbf x_b-\mu_b)^T\Lambda_{bb}(\mathbf x_b-\mu_b)\end{aligned}\tag{2.1}''')
st.markdown("&emsp;&emsp;展开以后有：")
st.latex(r'''\begin{aligned}&-\frac{1}{2}\mathbf x_{a}^T\Lambda_{aa}\mathbf x_{a} +\mathbf x_{a}^T\Lambda_{aa}\mu_a-\mathbf x_{a}^T\Lambda_{ab}(\mathbf x_b-\mu_b)+C \\ 
&= -\frac{1}{2}\mathbf x_{a}^T\Lambda_{aa}\mathbf x_{a} +\mathbf x_{a}^T\left\{\Lambda_{aa}\mu_a-\Lambda_{ab}(\mathbf x_b-\mu_b)\right\}+C \\ 
&\sim -\frac{1}{2}\mathbf x^T\mathbf \Sigma^{-1}_{a|b}\mathbf x +\mathbf x^T\mathbf\Sigma^{-1}_{a|b}\mu+C \end{aligned}\tag{2.2}''')
st.markdown("&emsp;&emsp;公式$(2.2)$倒数第一二行的两个公式互相比较，发现元素可以一一对应上，因此得出结论：")
st.latex(r'''\begin{aligned} \Lambda_{aa}&=\Sigma^{-1}_{a|b},~\Sigma^{-1}_{a|b}\mu_{a|b}=\Lambda_{aa}\mu_a-\Lambda_{ab}(\mathbf x_b-\mu_b) \\
&\rightarrow \mathbf \Sigma_{a|b}=\Lambda_{aa}^{-1}\\
\mu_{a|b}&=\mathbf \Sigma_{a|b}\left\{\Lambda_{aa}\mu_a-\Lambda_{ab}(\mathbf x_b-\mu_b)\right\}\\
&=\mu_a-\Lambda_{aa}^{-1}\Lambda_{ab}(\mathbf x_b-\mu_b) \\
\end{aligned} \tag{2.3}''')
st.markdown("&emsp;&emsp;从公式$(2.3)$的结果可知，$\mu_{a|b}$和$\mathbf \Lambda_{aa}$和$\mathbf \Lambda_{ab}$相关，所以接下来就推导这两个元素。注意，上式子，为了更好地推导$\Sigma_{a|b}$,我们先引出分块矩阵的逆：")
st.latex(r'''\begin{aligned}\left( \begin{array} { c c c c } { \mathbf A } & { \mathbf B }  \\ { \mathbf C} & {\mathbf D}  \end{array} \right)^{-1}&=\left( \begin{array} { c c c c } { \mathbf M } & { \mathbf {-MBD^{-1}} }  \\ { \mathbf {-D^{-1}CM}} & {\mathbf {D^{-1}+D^{-1}CMB D^{-1}}}  \end{array} \right) \\
\mathbf M~&=~(\mathbf A-\mathbf {BD^{-1}C})^{-1}\end{aligned} \tag{2.4}''')
st.markdown("&emsp;&emsp;同时，根据精度矩阵的定义，我们有:")
st.latex(r'''\mathbf \Sigma=\left( \begin{array} { c c c c } { \mathbf \Sigma_{aa} } & { \mathbf \Sigma_{ab} }  \\ { \mathbf \Sigma_{ba}} & {\mathbf \Sigma_{bb}}   \end{array} \right)^{-1}=\left( \begin{array} { c c c c } { \mathbf \Lambda_{aa} } & { \mathbf \Lambda_{ab} }  \\ { \mathbf \Lambda_{ba}} & {\mathbf \Lambda_{bb}}   \end{array} \right)''')
st.markdown("&emsp;&emsp;利用上述分块矩阵的逆的公式$(2.4)$，我们可以得到:")
st.latex(r'''\begin{aligned}\mathbf \Lambda_{aa}&= (\mathbf \Sigma_{aa}- \mathbf \Sigma_{ab} \mathbf \Sigma_{bb}^{-1} \mathbf \Sigma_{ba})^{-1}  ~\textcolor{red}{关键元素}\\
\mathbf \Lambda_{ab}&=-(\mathbf \Sigma_{aa}- \mathbf \Sigma_{ab} \mathbf \Sigma_{bb}^{-1} \mathbf \Sigma_{ba})^{-1}\mathbf \Sigma_{ab}\Sigma_{bb}^{-1}\end{aligned}''')

st.markdown("&emsp;&emsp;将此结果带入回公式$(2.3)$于是便有：")
st.latex(r'''\begin{aligned}\mathbf \mu_{a|b}&=\Sigma_{a|b}\left\{\Lambda_{aa}\mu_a-\Lambda_{ab}(\mathbf x_b-\mu_b)\right\}\\
    &=\mu_a-\Lambda_{aa}^{-1}\Lambda_{ab}(\mathbf x_b-\mu_b) \\
&=\mu_a+\mathbf \Sigma_{ab}\mathbf \Sigma_{bb}^{-1}(\mathbf x_b-\mu_b) \\
\end{aligned} \tag{2.5}''')
st.latex(r'''\begin{aligned}\mathbf \Sigma_{a|b}&=(\mathbf \Sigma_{aa}- \mathbf \Sigma_{ab} \mathbf \Sigma_{bb}^{-1} \mathbf \Sigma_{ba})\end{aligned}\tag{2.6}''')
st.markdown("&emsp;&emsp;得到了给定$\mathbf x_b$的条件下$\mathbf x_{a|b}$的均值向量和协方差矩阵，那么此条件概率密度函数便可以被唯一确认。\
值得注意的是，由公式$(2.5)$可知，:red[$\mu_{a|b}$是关于$\mathbf x_b$的线性函数]，因为$\mu_a$和$\mu_b$是确认的，而$\mathbf \Sigma_{ab}$,$\mathbf \Sigma_{bb}$也是确认的，所以$\mu_{a|b}$随着随机变量$\mathbf x_{b}$的变化而线性变化。")
with st.expander("$(2.2)$小章各变量之间的关系总结如下："):
    st.latex(r'''\begin{aligned}\mathbf \mu_{a|b}&
    =\mu_a+\mathbf \Sigma_{ab}\mathbf \Sigma_{bb}^{-1}(\mathbf x_b-\mu_b) \\
    \mathbf \Sigma_{a|b}&=(\mathbf \Sigma_{aa}- \mathbf \Sigma_{ab} \mathbf \Sigma_{bb}^{-1} \mathbf \Sigma_{ba}) \\
    \mathbf \mu_{b|a}&=\mu_b+\mathbf \Sigma_{ba}\mathbf \Sigma_{aa}^{-1}(\mathbf x_a-\mu_a) \\
    \mathbf \Sigma_{b|a}&=(\mathbf \Sigma_{bb}- \mathbf \Sigma_{ba} \mathbf \Sigma_{aa}^{-1} \mathbf \Sigma_{ab})\\
    \Lambda_{aa}&= (\mathbf \Sigma_{aa}- \mathbf \Sigma_{ab} \mathbf \Sigma_{bb}^{-1} \mathbf \Sigma_{ba})^{-1}\\
    \Lambda_{bb}&= (\mathbf \Sigma_{bb}- \mathbf \Sigma_{ba} \mathbf \Sigma_{aa}^{-1} \mathbf \Sigma_{ab})^{-1}\end{aligned}\tag{a*}''')

st.markdown("##### 2.2.1 :blue[趣味环节]")
st.markdown("可以拖动滑动条来选择二元高斯分布各维度的均值~")
st.markdown("```为方便实验，二元高斯分布的协方差已经提前固定了噢，没法变动~```")
x_mu=st.slider("选择x维度变量的均值",-10.0,10.0,0.0,0.1)
y_mu=st.slider("选择y维度变量的均值",-10.0,10.0,0.0,0.1)
mu = np.array([x_mu, y_mu])
sigma = np.array([[1, 0.5], [0.5, 1]])

x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        vec = np.array([X[i, j], Y[i, j]])
        Z[i, j] = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.dot((vec - mu), np.linalg.inv(sigma)), (vec - mu).T))

fig = go.Figure(data=[
    go.Surface(x=x, y=y, z=Z, colorscale='Viridis')
])
fig.update_layout(
    title='二元正态分布',
    xaxis_title='x',
    yaxis_title='y',
)
st.plotly_chart(fig, use_container_width=True)

y_observation=st.slider("请选择y维度变量的取值",-10.0,10.0,0.0,0.1)
xmu_condition_y=x_mu+2/3*(y_observation-y_mu)

# 生成正态分布数据
y1 = norm.pdf(x, xmu_condition_y, 1)  # 均值为0，标准差为1的正态分布
# 创建 Plotly 曲线图
fig1 = go.Figure(data=go.Scatter(x=x, y=y1, mode='lines'))
fig1.update_layout(
    title='条件概率密度函数',
    xaxis_title='x',
    yaxis_title='y',
    template='plotly_white'  # 图表样式选择
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("#### :blue[2.3边缘高斯分布]")
st.latex(r'''p(\mathbf x_a)=\int p(\mathbf x_a,\mathbf x_b)\rm d\mathbf x_b''')
st.markdown("&emsp;&emsp;由公式$(2.1)$可知，指数项的展开可以写成如下：")
st.latex(r'''\begin{aligned}& -\frac{1}{2}\mathbf x_{b}^T\Lambda_{bb}\mathbf x_{b} +\mathbf x_{b}^T\left\{\Lambda_{bb}\mu_a+\Lambda_{ba}(\mathbf x_a-\mu_a)\right\}+C \end{aligned} \tag{2.7}''')
st.markdown("&emsp;&emsp;为了配方，回顾一下以前的一元二次方程$ax^2+bx+c$可以变成$a(x-k)^2+m$的形式，其中$k=-\cfrac{b}{2a},m=c-\cfrac{b^2}{4a}$")
st.markdown("&emsp;&emsp;推广到矩阵形式，有：")
st.latex(r'''\begin{aligned}& \mathbf x^TA\mathbf x +\mathbf x^TB =  (\mathbf x-k)^TA (\mathbf x- k) +m \\&=\mathbf x^TA\mathbf x-2\mathbf x^TAk+k^TAk+m\\& \Rightarrow -2\mathbf x^TAk=\mathbf x^TB ,k^TAk+m=0 \\ &\Rightarrow k=-\frac{1}{2}A^{-1}B,m=\frac{1}{4}B^TAB \end{aligned}''')
st.markdown("将此结论带入到上式，有：")
st.latex(r'''\begin{aligned}& -\frac{1}{2}\mathbf x_{b}^T\Lambda_{bb}\mathbf x_{b} +\mathbf x_{b}^T\left\{\Lambda_{bb}\mu_a+\Lambda_{ba}(\mathbf x_a-\mu_a)\right\}+C \\ \
 &=\underbrace{-\frac{1}{2}(\mathbf x_{b}-(\mu_b-\Lambda_{bb}^{-1}\Lambda_{ab}(\mathbf x_b-\mu_b))\Lambda_{bb}(\mathbf x_{b}-(\mu_b-\Lambda_{bb}^{-1}\Lambda_{ab}(\mathbf x_b-\mu_b))-}_{E_1}\\ &\underbrace{\frac{1}{2}(\mathbf x_a-\mu_a)^T(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})(\mathbf x_a-\mu_a)  }_{E_2}\\ &=E_1+E_2 \
\end{aligned} \tag{2.8}''')
st.markdown("&emsp;&emsp;通过配方我们已经将变量$\mathbf x_a$和$\mathbf x_b$写成了两部分，接下来我们通过积分联合概率密度函数得到$\mathbf x_a$的边缘概率密度函数：")
st.latex(r'''\begin{aligned}&p(\mathbf x_a,\mathbf x_b)=\frac{1}{(2\pi)^D}\frac{1}{|\mathbf\Sigma|^{1/2}}\mathbf {exp}\left\{E_1+E_2 \right\} \\
  &p(\mathbf x_a)=\int p(\mathbf x_a,\mathbf x_b)\rm d\mathbf x_b=\frac{1}{(2\pi)^D}\frac{1}{|\mathbf\Sigma|^{1/2}}\int \mathbf {exp}\left\{E_1\right\}\mathbf {exp}\left\{E_2\right\}\rm d\mathbf x_b \\
&p(\mathbf x_a)=\frac{1}{(2\pi)^D}\frac{1}{|\mathbf\Sigma|^{1/2}}\mathbf {exp}\left\{E_2\right\}\int \mathbf {exp}\left\{E_1\right\}\rm d\mathbf x_b \\
&p(\mathbf x_a)=\frac{1}{(2\pi)^{D_a}}\frac{|\Lambda_{bb}^{-1}|^{1/2}}{|\mathbf\Sigma|^{1/2}}\mathbf {exp}\left\{E_2\right\}\end{aligned} \tag{2.9}''')
st.markdown("&emsp;&emsp;进一步化简，根据分块矩阵的行列式：")

st.latex(r'''\begin{aligned}\begin{vmatrix}A & B \\
   C & D\\  \end{vmatrix}&=|A||D-CA^{-1}B|  \\
       \Rightarrow |\mathbf \Sigma|&=\begin{vmatrix}\mathbf \Sigma_{aa} & \mathbf \Sigma_{ab} \\ \mathbf \Sigma_{ba} & \mathbf \Sigma_{bb}\\  \end{vmatrix}\\
        &=|\Sigma_{aa}||\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{aa} | \end{aligned} \tag{2.10}''')

st.markdown("&emsp;&emsp;利用$a^*$中的结论，可以得到$|\mathbf \Lambda_{bb}^{-1}|=|\mathbf{\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{aa} }|$，故：")
st.latex(r'''\begin{aligned}\frac{\mathbf \Lambda_{bb}^{-1}|^{1/2}}{|\mathbf\Sigma|^{1/2}}&=\frac{1}{|\mathbf\Sigma_{aa}|^{1/2}}\\
\Rightarrow p(\mathbf x_a)&=\frac{1}{(2\pi)^{D_a}}\frac{1}{|\mathbf\Sigma_{aa}|^{1/2}}\mathbf {exp}\left\{E_2\right\} \end{aligned} \tag{2.11}''')
st.markdown("&emsp;&emsp;$\mathbf x_a$的均值向量和协方差矩阵如下：")
st.latex(r'''\mathbf E[\mathbf x_a]=\mu_a,\mathbf{cov}[\mathbf x_a]=(\mathbf\Lambda_{aa}-\mathbf \Lambda_{ab}\mathbf \Lambda_{bb}^{-1}\mathbf \Lambda_{ba})^{-1}=\mathbf {\Sigma_{aa}} \tag{2.12}''')

st.markdown("#### :blue[2.4高斯变量的贝叶斯理论]")
st.markdown("&emsp;&emsp;在$2.2$小节，我们学习了在知道联合概率密度的情况下，给定$\mathbf x_b$求解条件概率密度$p(\mathbf{x_a|x_b})$，通过概率密度函数指数项内的展开以及借助分块矩阵的逆，我们成功推导出了条件概率密度，并且发现了给定$\mathbf x_b$的条件下，$\mu_{a|b}$是关于随机变量$\mathbf x_b$的线性函数。")
st.markdown("&emsp;&emsp;在$2.3$小节，我们学习了在知道联合概率密度的情况下，给定$\mathbf x_b$求解边缘概率密度$p(\mathbf x_a)$，通过配方法和引入分块矩阵的行列式，我们也成功解决了问题。\
    在本小节，我们将学习知道边缘概率密度和条件概率密度的情况下，求解联合概率密度函数。")

st.markdown("&emsp;&emsp;首先给定边缘概率密度与条件概率密度，如下所示：")
st.latex(r'''\begin{aligned} &p(\mathbf x)=\mathcal{N}(\mathbf x|\mu,\mathbf\Lambda^{-1})\\ 
&p(\mathbf y|\mathbf x)=\mathcal{N}(\mathbf x|\mathbf A\mu+\mathbf b,\mathbf L^{-1}) \end{aligned} \tag{2.13}''')
st.markdown("&emsp;&emsp;我们将$\mathbf{x,y}$组合起来得到$$\mathbf z=\\begin{pmatrix} \mathbf {x} \\\ \mathbf {y}\end{pmatrix}$$，只考虑概率密度函数指数项有：")
st.latex(r'''\begin{aligned}\ln p(\mathbf z)&=\ln p(\mathbf x) + \ln p(\mathbf {y|x})\\
&=-\frac{1}{2}(\mathbf {x-\mu})^T\mathbf \Lambda(\mathbf {x-\mu)}\\
&-\frac{1}{2}(\mathbf {y-Ax-b})^T\mathbf L(\mathbf {y-Ax-b})  \\
&=-\frac{1}{2}{\mathbf {x^T (\Lambda+A^{T}LA )x}} -\frac{1}{2}{\mathbf {y^T Ly}}+\frac{1}{2}{\mathbf {y^T LAx}}+\frac{1}{2}{\mathbf {x^T A^T Ly}} \\
&=-\frac{1}{2}\begin{pmatrix} \mathbf x  \\\mathbf y \end{pmatrix}^T \left( \begin{array} { c c c c } { \mathbf {\Lambda +A^T LA} } & {- \mathbf {A^T L} }  \\ {- \mathbf {LA}} & {\mathbf L}   \end{array} \right) \begin{pmatrix} \mathbf x  \\\mathbf y \end{pmatrix} \\
&=-\frac{1}{2}\mathbf {z^T Rz}\end{aligned} \tag{2.14}''')
st.markdown("&emsp;&emsp;从公式$2.14$的结果可知，$\mathbf R$是协方差矩阵的逆，$\mathbf z$的协方差矩阵就是$\mathbf R^{-1}$，接下来只要确定均值向量就可以确定联合概率密度分布。")
st.markdown("&emsp;&emsp;提取出联合概率密度即公式$2.14$中和$\mathbf {x,y}$相关的线性项，则有：")
st.latex(r'''\begin{aligned}\begin{pmatrix} \mathbf x  \\\mathbf y \end{pmatrix}^T\begin{pmatrix} \mathbf {\Lambda \mu-A^T Lb}  \\ \mathbf {Lb} \end{pmatrix}&= \begin{pmatrix} \mathbf x  \\\mathbf y \end{pmatrix}^T \mathbf R^{-1}\mathbb E[\mathbf z] \\
&\Rightarrow \mathbb E[\mathbf z]=\mathbf R\begin{pmatrix} \mathbf {\Lambda \mu-A^T Lb}  \\ \mathbf {Lb} \\\end{pmatrix}\\
&=\begin{pmatrix} \mathbf {\mu}  \\ \mathbf {A\mu+b} \\\end{pmatrix} \end{aligned} \tag{2.15}''')
st.markdown("&emsp;&emsp;注：公式$2.15$中第一行的等价关系实际上是运用到了公式$2.1$的结论，即高斯概率密度函数展开以后的和$\mathbf x$有关的线性项形式为$\mathbf {x\Sigma \mu}$。")
st.markdown("&emsp;&emsp;最终得到的联合概率密度函数的均值和协方差矩阵如下：")
st.latex(r'''\begin{aligned} \mathbf {cov[\mathbf z]}&= \left( \begin{array} { c c c c } { \mathbf {\Lambda^{-1}} } & { \mathbf {\Lambda^{-1}A^T} }\
    \\ { \mathbf {A\Lambda^{-1}L}} & {\mathbf {L^{-1}+A\Lambda^{-1}A^T}}  \end{array} \right)=\mathbf R^{-1} \\
        \mathbb E[\mathbf z] &=\begin{pmatrix} \mathbf {\mu}  \\ \mathbf {A\mu+b} \\\end{pmatrix} \end{aligned} \tag{2.16}''')
st.markdown("&emsp;&emsp;此外，根据联合概率密度的协方差矩阵$\mathbf {cov[\mathbf z]}$和公式$2.3$，我们可以反过来求给定$\mathbf y$的条件下,$\mathbf x$的均值和协方差矩阵。")
st.latex(r''' ''')

