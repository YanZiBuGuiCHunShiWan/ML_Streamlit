import streamlit as st


def Markov_Chain_Monte_Carlo():
    st.markdown("# :blue[马尔可夫链蒙特卡洛方法]")
    st.markdown("&emsp;&emsp;马尔可夫链蒙特卡洛方法俗称$\\text {MCMC}$,第一个$\\text {MC}$是指马氏链，我们利用马氏链采样得到服从某个分布的样本，第二个$\\text {MC}$指的是蒙特卡洛方法，即采样得到的样本基于蒙特卡洛思想进行数值计算，因此其是两部分工作的总和。")
    st.markdown("&emsp;&emsp;以下是一个为什么能利用马氏链采样的直观案例，假设有一个马氏链，其状态集合是$\{A,B,C \}$，状态转移矩阵记作$M$,且矩阵中元素$m_{ij}≐P(x_{t+1}=j|x_t=i)$，即当前时刻是状态$i$，下一个时刻转移到状态$j$的概率，因此这个状态转移矩阵每一行之和为1。（有些教材定义的是每一列之和为1）")
    st.latex(r'''M=\left(\begin{array}{lll}
    m_{A A} & m_{A B} & m_{A C} \\
    m_{B A} & m_{B B} & m_{B C} \\
    m_{C A} & m_{C B} & m_{C C}
    \end{array}\right)''')
    st.markdown("&emsp;&emsp;如果转换矩阵$M$不随时间变化而变化，我们就称这样的马尔可夫链是齐次的$\\text {(homogeneous)}$或者说时齐的$\\text {(time-homogeneous)}$、平稳的（稳定的）$\\text {(stationary)}$。我们研究的主要是齐次马尔可夫链。齐次马尔可夫链的状\
        态转移矩阵有一个非常重要的性质。假设状态转移矩阵为：")
    st.latex(r'''M=\left(\begin{array}{lll}
    0.3 & 0.2 & 0.5 \\
    0.15 & 0.33 & 0.52 \\
    0.4 & 0.1 & 0.5
    \end{array}\right)''')
    st.markdown("&emsp;&emsp;下面计算一下矩阵$M$的$n$次幂：")
    st.latex(r'''
             \begin{aligned}M^{2}&=\left(\begin{array}{ccc}
0.32 & 0.176 & 0.504 \\
0.303 & 0.191 & 0.506 \\
0.335 & 0.163 & 0.502
\end{array}\right) \\
    M^{3}&=\left(\begin{array}{lll}
0.324 & 0.172 & 0.504 \\
0.322 & 0.174 & 0.504 \\
0.326 & 0.171 & 0.503
\end{array}\right) \\
    &\cdots \\
        M^{10}&=\left(\begin{array}{lll}
0.325 & 0.172 & 0.503 \\
0.325 & 0.172 & 0.503 \\
0.325 & 0.172 & 0.503
\end{array}\right)\end{aligned}
             ''')
    st.markdown("&emsp;&emsp;随着次幂的数值的增大，矩阵$M^{n}$的数值也变得稳定，且每一行都一样，这是转移矩阵非常重要的一个特性。:red[（并不是任意一个转移矩阵的$n$次幂都会变成一个稳定的矩阵，是有一定条件的。）]\
        $\\text {MCMC}$的采样正是基于这一性质实现的。另外，我们发现，当$M^n$稳定后，服从任意分布的变量经$M^n$转移后都服从同一个分布，这个分布就是$M^n$中行向量代表的那个分布。")
    st.markdown("&emsp;&emsp;例如分布$\pi_{0}(x)=(0.1,0.3,0.6)$，那么有：")
    st.latex(r'''\pi^{\prime}(x)=\pi_{0}(x) M^{n}=(0.325,0.172,0.503)''')
    st.markdown("&emsp;&emsp;记达到稳定后的$M$矩阵为$M^*$，则任意的$\pi(x)$与$M^*$相乘都是$M^*$的行向量。假设对于能够达到稳定状态的转移矩阵$M$，它对应的特殊分布为$\pi$，那么此时一定有：")
    st.latex(r'''\pi=\pi M''')
    