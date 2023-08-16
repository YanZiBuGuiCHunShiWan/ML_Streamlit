import streamlit as st

st.title("近似推断")
st.sidebar.markdown("# 近似推断")
st.markdown("## :blue[前置知识：变分法]")

st.markdown("## :blue[6.1 变分推断]")

####################################################6.1.1#############################################################
st.markdown("### :blue[6.1.1 Factorized distributions]")
st.markdown("&emsp;&emsp; 我们希望变分分布$q(\mathbf Z)$和完全分布越相似越好，通过$KL$散度来衡量二者的相似性，我们可以得到：")
st.latex(r'''\begin{aligned}\mathcal L(q)&=\int \prod_i q_i\{\ln p(\mathbf{X,Z})-\sum_i \ln q_i \}\rm d \mathbf Z \\
    &=\int q_j \{ \int \ln p(\mathbf{X,Z}) \prod_{i \neq j} q_i \rm d \mathbf Z_i \} \rm d \mathbf {Z_j} - \int q_j \ln q_j \rm d \mathbf Z_j + C \\
    &=\int q_j\ln {\tilde p(\mathbf {X,Z_j} )}\rm d \mathbf {Z}_j-{\int q_j \ln q_j} \rm d \mathbf Z_j + C\\
    &=-KL(q_j||\tilde P(\mathbf{X,Z}))+C\end{aligned} \tag{6.1}''')

st.markdown("&emsp;&emsp;其中：")
st.latex(r'''\begin{aligned}\ln {\tilde p(\mathbf{X,Z_j})} &=\int \ln p(\mathbf{X,Z}) \prod_{i \neq j} q_i \rm d \mathbf Z_i \\
    &=\mathbb E_{q_i(z_i):i \neq j}[\ln {p(\mathbf{X,Z})}]+c\end{aligned} \tag{6.2}''')
st.markdown("&emsp;&emsp;式子$6.1$中，$-KL$恒小于等于0，当$\ln q_j^*(\mathbf Z_j)=\mathbb E_{q_i(z_i):i\\not =j}[\ln {p(\mathbf{X,Z})}]+c$时，公式取得最大值。进一步，有：")
st.latex(r'''\begin{aligned}q_j^*(\mathbf Z_j)=\exp\{ {\mathbb E_{q_i(z_i):i\not =j}[\ln {p(\mathbf{X,Z})}]+c} \}\end{aligned} \tag{6.3} ''')
st.markdown("&emsp;&emsp;为确保概率密度合法，有：")
st.latex(r'''\begin{aligned}q_j^*(\mathbf Z_j)=\frac{\exp({\mathbb E_{q_i(z_i):i\not =j}[\ln {p(\mathbf{X,Z})}]})}{\int  \exp({\mathbb E_{q_i(z_i):i\not =j}[\ln {p(\mathbf{X,Z})}]}) \rm d \mathbf Z_j}\end{aligned} \tag{6.4}''')

st.markdown("&emsp;&emsp;在实际的求解过程中，我们会先初始化一组$q_1,q_2,....,q_m$，每次固定其他的$q_{-j}$来更新$q_j$。")


######################################################6.1.2############################################################
st.markdown("### :blue[6.1.2 Factorized approximations]")
st.markdown("&emsp;&emsp;接下来看一个分解近似(factorized approximations)的实例，我们用因子分解高斯来近似一个二元高斯分布。给定一个二元高斯分布$p(\mathbf z) \
    =\mathcal N(\mathbf {z|\mu,\Lambda^{-1}})$,其中$\mathbf z$由两个相关的变量$z_1,z_2$构成，且对应的均值和方差如下：")
st.latex(r'''\begin{aligned}\mu=\begin{pmatrix} \mu_1 \\\mu_2 \end{pmatrix}, \mathbf \Lambda=\begin{pmatrix} \Lambda_{11} & \Lambda_{12}  \\\Lambda_{21} & \Lambda_{22}  \end{pmatrix}\end{aligned} \tag{6.5}''')
st.markdown("&emsp;&emsp;我们令$q(\mathbf z)=q(z_1)q(z_2)$，然后根据式子$6.3$可以得到$q^*_1(z_1)$的表达式如下：")
st.latex(r'''\begin{aligned} \ln {q^*_1(z_1)}&=\mathbb E_{z2}[\ln p(\mathbf z)]+c \\
    &=\mathbb E_{z2} [-\frac{1}{2}(z_1-\mu_1)^2\Lambda_{11}-(z_1-\mu_1)\Lambda_{12}(z_2-\mu_2)]+c\\
    &=-\frac{1}{2}\Lambda_{11}z^2_1+z_1\{\mu_1\Lambda_{11}-\Lambda_{12}(\mathbb E[z_2]-\mu_2) \}+c \end{aligned} \tag{6.6}''')
st.markdown("&emsp;&emsp;公式$6.6$的结果就形如$2.1$第一行等式右边的公式，所以我们可以直接得到结论，即$q_1(z_1)$是一个高斯分布，其均值方差如下：")
st.latex(r'''\begin{aligned} m_1&=\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(\mathbb E[z_2]-\mu_2) \\
    q^*(z_1)&=\mathcal N(z_1|m,\Lambda_{11}^{-1})\end{aligned} \tag{6.7}''')
st.markdown("&emsp;&emsp;同理，我们可以得到$q_2^*(z_2)$的分布：")
st.latex(r'''\begin{aligned}q_2^*(z_2)&=\mathcal N(z_2|m_2,\Lambda_{22}^{-1})\\
    m_2&=\mu_2-\Lambda_{22}^{-1}\Lambda_{21}(\mathbb E[z_1]-\mu_1)\end{aligned}\tag{6.8}''')
st.markdown("&emsp;&emsp;:red[若$p(\mathbf z)$是非奇异分布，那么最终一定有$m_1=\mu_1,m_2=\mu_2$]。证：")
st.latex(r'''\begin{aligned} m_1&=\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(\mathbb E[z_2]-\mu_2)\\
    &=\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(\mu_2-\Lambda_{22}^{-1}\Lambda_{21}(\mathbb E[z_1]-\mu_1)-\mu_2)\\
        &=\mu_1+\Lambda_{11}^{-1}\Lambda_{12}\Lambda_{22}^{-1}\Lambda_{21}(m_1-\mu_1)\end{aligned} \tag{6.9}''')
st.markdown("&emsp;&emsp;当且仅当$m_1=\mu_1$时，等式恒成立，同理可得$m_2=\mu_2$。若$p(\mathbf z)$是奇异分布，那么，根据$\mathbb E[z_1],\mathbb E[z_2]$初始值\
    的不同 ，最终$m_1,m_2$有可能等于$\mu_1,\mu_2$，也有可能不等。")
st.markdown("&emsp;&emsp;在以上公式的推导过程中，我们都是基于最小化$KL(q||p)$的思想进行推理的，那如果反过来最小化$KL(p||q)$的思想进行推理，那么得到的结果又会有什么不同呢？")
st.markdown("&emsp;&emsp;我们先看优化原本的$KL(q||p)$能得到什么样的结论。当$p(\mathbf z) \\rightarrow 0$时，为了使得$KL(q||p)$趋向于$0$，我们只能让$q(\mathbf z) \\rightarrow 0$，如果不趋于$0$\
    ，那么$KL(q||p)$就趋向于无穷大，这就违背了优化$KL(q||p)$的出发点。当$p(\mathbf z) \\rightarrow const, const \\neq 0$，则必有$q(\mathbf z) \\rightarrow p(\mathbf z) or 0$。")
st.markdown("&emsp;&emsp;情况二：优化$KL(p||q)$，当$p(\mathbf z)\\rightarrow const,const \\neq 0$时，$KL(p||q)=\int p(\mathbf z)\ln \\frac{p(\mathbf z)}{q(\mathbf z)} \\rm d \mathbf Z $，必有$q(\mathbf z) \\rightarrow p(\mathbf z)$；\
    若$p(\mathbf z) \\rightarrow 0$，则$q(\mathbf z) \\rightarrow p(\mathbf z) or 0$。")
st.image("src/6_1.png")
st.markdown("&emsp;&emsp;上图来自于prml第十章图10.3，其表达的信息如下：(a)图的蓝色等高线是混合高斯分布的概率密度等高线，可见该混合分布有两个峰，当我们用\
    $KL(p||q)$去近似该混合分布时，得到的近似分布的概率密度等高线是红色的椭圆曲线，并且覆盖了原有分布的概率密度等高线，因为由上述情况二的结论有$p(\mathbf z)$不趋向于0时，$q(\mathbf z)$ \
        一定会趋向于$p(\mathbf z)$。")
st.latex(r'''\begin{aligned}  \end{aligned}''')

############################################################6.1.3##########################################################
st.markdown("### :blue[6.1.3 一元高斯模型]")
st.markdown("&emsp;&emsp;接下来介绍如何使用$\\text{factorized variational approximation}$来近似一元高斯分布，我们的目的是:red[推断后验分布的均值$\mu$和精度$\\tau$]。\
    给定了观察数据$\mathcal D=\{ x_1,...,x_N \}$,假设观测数据是相互独立的，那么我们可以写出似然函数：")
st.latex(r'''\begin{aligned} p(\mathcal D|\mu,\tau)={(\frac{\tau}{2\pi})}^{\frac{N}{2}} \exp \left\{ -\frac{\tau}{2}\sum_{n=1}^{N}(x_n-\mu)^2  \right\}\end{aligned}''')