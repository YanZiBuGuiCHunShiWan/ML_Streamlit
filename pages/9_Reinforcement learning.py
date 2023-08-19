import streamlit as st

def reinforcement_foundation():
    st.sidebar.markdown("## 10.1 强化学习简介")
    st.sidebar.markdown("## 10.2 强化学习的最终目的")
    st.sidebar.markdown("### &emsp;10.2.1 策略和价值函数")
    st.sidebar.markdown("### &emsp;10.2.2 最优策略和最优价值函数")
    st.sidebar.markdown("## 10.3 动态规划")
    st.sidebar.markdown("### &emsp;10.3.1 策略评估")
    st.sidebar.markdown("#### &emsp;&emsp;10.3.1.1迭代策略评估")
    st.sidebar.markdown("### &emsp;10.3.2 策略改进")
    st.sidebar.markdown("### &emsp;10.3.2 策略迭代")
    st.sidebar.markdown("### &emsp;10.3.3 价值迭代")
    st.markdown("### :blue[10.1强化学习简介]")
    st.markdown("### :blue[10.2强化学习的最终目的]")
    st.latex(r'''\begin{aligned} G_t&=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+ ... +\gamma^{T-1} R_{T} \\
        &=\sum_{i=0}\gamma^{i} R_{t+i+1}\end{aligned}''')
    st.markdown("&emsp;&emsp;折扣率$\gamma$决定了未来收益的现值，如果$\gamma$接近于$0$则说明智能体目光短浅，因为后效的收益接近于$0$。若$\gamma$趋向于$1$，则表明\
        智能体有远见，能考虑长远收益。")
    st.image("src/强化学习1_01.jpg")
    st.markdown("#### :blue[10.2.1 策略和价值函数]")
    st.markdown("&emsp;&emsp;价值函数是关于状态的函数，我们需要评估当前智能体在给定状态下的一个优劣，而‘优劣’是用:blue[未来预期的收益]来定义的。简单说来就是回报的期望值，\
    我们把策略$\pi$下状态$s$的价值函数记作$v_{\pi}(s)$，即从状态$s$开始，智能体按照策略$\pi$进行决策获得的回报的期望。正式定义$v_{\pi}(s)$如下：")
    st.latex(r'''\begin{aligned} v_{\pi}(s)≐\mathbb E_{\pi}[G_t|S_t=s]=\mathbb E_{\pi}[\sum_{i=0}\gamma^{i} R_{t+i+1}|S_t=s],s\in \mathcal S\end{aligned} \tag{10.1}''')
    st.markdown("&emsp;&emsp;类似地，我们把在策略$\pi$下状态$s$时采用动作$a$所获得的回报期望记作$q_{\pi}(s,a)$，其定义如下：")
    st.latex(r'''\begin{aligned}q_{\pi}(s,a)≐\mathbb E_{\pi}[G_t|S_t=s,A_t=a]=\mathbb E_{\pi}[\sum_{i=0}\gamma^{i} R_{t+i+1}|S_t=s,A_t=a]\end{aligned} \tag{10.2}''')
    st.markdown("&emsp;&emsp;显然，$v_{\pi}(s)$和$q_{\pi}(s,a)$有密切关系，由上图可知，状态$s$下可采取的动作$s$有$|\mathcal S|$个，每个动作$a$都对应于一个$q_{\pi}(s,a)$,\
        所以有:red[$v_{\pi}(s)=\sum_{a\in \mathcal A} \pi(a|s)q_{\pi}(s,a)$]。")
    st.markdown("&emsp;&emsp;在强化学习与动态规划中，价值函数实际上是满足某种递推关系的，从图中可以看到$q_{\pi}(s,a)$和$v_{\pi}(s)$实际上也是满足某种关系的，我们来推导一下：")
    st.latex(r'''\begin{aligned} q_{\pi}(s,a)&=\sum_{s',r} p(s',r|s,a) [r+\gamma \mathbb E_{\pi}[G_{t+1}|S_{t+1}=s']] \\
            &=\sum_{s',r} p(s',r|s,a) [r+\gamma v_{\pi}(s')] \\
                &=\sum_{s',r} p(s',r|s,a) [r+\gamma \sum_{a'\in \mathcal A} \pi(a'|s')q_{\pi}(s',a')] \end{aligned} \tag{10.3}''')
    st.markdown("&emsp;&emsp;$10.3$的意思是（结合图看），动作$a$产生状态$s'$会有一个收益$r$然后得到衰减了的$\gamma v_{\pi}(s)$，但这个$r$实际上是不固定的，有很多个可能性，所以要对$r$加权平均，而$s'$取值也有$|\mathcal S|$个，所以也要加权平均。\
                最终的$v_{\pi}(s)$的递推式表达如下：")
    st.latex(r'''\begin{aligned} v_{\pi}(s) &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma \mathbb E_{\pi}[G_{t+1}|S_{t+1}=s']]\\
        &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma v_{\pi}(s')]\end{aligned} \tag{10.4}''')
    st.markdown("&emsp;&emsp;式子$10.3$和$10.4$被称作:red[贝尔曼期望方程]，其分别描述了:red[动作价值和后继动作价值之间的关系]以及:red[状态价值和后继状态价值之间的关系]。")
    with st.expander("💡:blue[Example1]💡"):
        st.markdown("接下来是一个计算实例。")
    st.markdown("#### :blue[10.2.2 最优策略和最优价值函数]")
    st.markdown("&emsp;&emsp;强化学习的最终目的就是去找到最优的$\pi$使得回报期望最大，而算法中能优化的也只有$\pi$。我们将使得价值函数达到最大，我们将最优状态价值函数和最优动作价值函数定义如下：")
    st.latex(r'''\begin{aligned} v_{*}(s)&≐ \max_{\pi} v_{\pi}(s) \\
        q_{*}(s,a)&≐\max_{\pi} q_{\pi}(s,a)\end{aligned} \tag{10.5}''') 
    st.markdown("&emsp;&emsp;尽管最优策略不止一个，但是我们统一表示如下，公式含义是$\pi_{*}$可以使得$v_{\pi}(s)$达到最大也可以使$q_{\pi}(s,a)$达到最大。")
    st.latex(r'''\pi^*=\argmax_{\pi} v_{\pi}(s)=\argmax_{\pi} q_{\pi}(s,a) \tag{10.6}''')
    st.markdown("&emsp;&emsp;最优策略下的状态价值函数一定等于这个状态下最优动作的期望回报，$v_*$的贝尔曼最优方程如下：")
    st.latex(r'''\begin{aligned} v_{*}(s)&=\max_{a \in \mathcal A} q_{\pi^*}(s,a) \\
        &=\max_a \sum_{s',r} p(s',r|s,a) [r+\gamma v_{\pi}(s')]\end{aligned} \tag{10.7}''')
    st.markdown("&emsp;&emsp;$q_*$的贝尔曼最优方程如下：")
    st.latex(r'''\begin{aligned}q_{*}(s,a)=\sum_{s',r} p(s',r|s,a) [r+\gamma \max_{a'\in \mathcal A}q_{*}(s',a')] \end{aligned} \tag{10.8}''')
    st.markdown("### :blue[10.3 动态规划]")
    st.markdown("#### :blue[10.3.1 策略评估]")
    st.markdown("&emsp;&emsp;我们要计算出状态价值函数来才能进一步评估策略的好坏，因此接下来介绍如何计算出具体的状态价值函数。我们的条件是：给定$MDP$,即我们知道\
        $p(s',r|s,a)$和策略$\pi$的条件下求$v_{\pi}(s)$的表达式。由先前$10.4$可知：")
    st.latex(r'''\begin{aligned}v_{\pi}(s)&=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma v_{\pi}(s')] \\
        &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} rp(s',r|s,a)+\gamma \sum_{a \in \mathcal A}\pi(a|s) \sum_{s',r}p(s',r|s,a)v_{\pi}(s') \\
            &=\underbrace {\sum_{a \in \mathcal A}\pi(a|s)\sum_{r} rp(r|s,a)}_{\text{\textcircled a}}+\underbrace{\gamma \sum_{a \in \mathcal A}\pi(a|s)\sum_{s'} p(s'|s,a)v_{\pi}(s')}_{\text{\textcircled b}}\end{aligned}''')
    st.markdown("##### :blue[10.3.1.1 迭代策略评估]")
    st.markdown("&emsp;&emsp;解析解虽然能写出来，但是直接求解时间复杂度太高了，因此通常用数值解来求解最优策略，我们的目标是求出：")
    st.latex(r'''V_{\pi}(s)=\begin{pmatrix} v_*(s_1)  \\ {\vdots} \\ v_*(|\mathcal S|) \end{pmatrix}''')
    st.markdown("&emsp;&emsp;在迭代的策略评估中，我们初始化一组序列$\{ v_k\}_{k=1}^{\infin}$，然后按照如下方程进行更新：")
    st.latex(r'''\begin{aligned}v_{k+1}(s)&≐\mathbb E_{\pi}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s] \\
        &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma v_{k}(s')]\end{aligned} \tag{10.9}''')
    st.markdown("&emsp;&emsp;最终一定会收敛到$v_{\pi}$，具体就不在此证明了。")
    
    st.markdown("#### :blue[10.3.2 策略改进]")
    st.markdown("&emsp;&emsp;当计算出了价值函数，我们就能评估策略的好坏，但对于某个状态，我们想知道是否应该选择不同于给定的策略的动作$a \\neq \pi(s)$，因为我们不清楚选择其他策略以后得到的结果会更好还是更坏。\
        最直观的办法是，直接计算$v_{\pi}(s)$和新策略对应的$v_{\pi '}(s)$，比较二者大小就知道策略好坏，但是实际情况下计算新的状态函数是要消耗资源的，有没有其他的办法能更简洁地判断呢？")
    st.markdown("&emsp;&emsp;我们有:blue[策略改进定理]能更简洁地告诉我们如何改进策略。内容如下：如果$\pi$和$\pi '$是两个确定的策略，如果对于任意$s \in \mathcal S$，我们有：")
    st.latex(r'''\begin{aligned} q_{\pi}(s,\pi '(s)) \geq v_{\pi}(s) \end{aligned}''')
    st.markdown("&emsp;&emsp;那么策略$\pi '$一定不亚于$\pi$，即：")
    
    st.latex(r'''\begin{aligned} v_{\pi '}(s) \geq v_{\pi}(s) \\\end{aligned}''')
    with st.expander("证明过程如下"):
        st.latex(r'''\begin{aligned}v_{\pi}(s)&\leq q_{\pi}(s,\pi'(s)) \\
            &=\mathbb E[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t=s,A_t=\pi'(s)] \\
                &=\mathbb E_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t=s]\end{aligned}''')
        st.markdown("&emsp;&emsp;又$v_{\pi}(S_{t+1}) \leq q_{\pi}(s,\pi'(S_{t+1})) $，故上式可以缩放成如下：")
        st.latex(r'''\begin{aligned}v_{\pi}(s)&\leq \mathbb E_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1},\pi'(S_{t+1}))|S_t=s] \\
            &=\mathbb E_{\pi'}[R_{t+1} + \gamma \mathbb E_{\pi'}[R_{t+2}+\gamma v_{\pi}(S_{t+2})|S_{t+1}]|S_t=s] \\
                &=\mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2}+\gamma^2 v_{\pi}(S_{t+2})|S_t=s]\end{aligned}''')
        st.markdown("&emsp;&emsp;上式之所以成立是因为$\pi'$是一个确定性的策略，所以对中括号内的元素求期望这个操作可以忽略。接着不断按照上式规律进行缩放得到如下结果：")
        st.latex(r''' \begin{aligned} v_{\pi}(s)&\leq \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2}+\gamma^2 v_{\pi}(S_{t+2})|S_t=s] \\
            &\leq \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2}+\gamma^2 v_{\pi}(S_{t+2}) + \gamma^3 v_{\pi}(S_{t+3}) |S_t=s] \\
                &\vdots \\
                    &\leq \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2}+\gamma^2 v_{\pi}(S_{t+2}) + \gamma^3 v_{\pi}(S_{t+3}) + \gamma^4 v_{\pi}(S_{t+4}) + \cdots |S_t=s] \\
                        &=v_{\pi'}(s)\end{aligned}''')
        
    st.markdown("&emsp;&emsp;既然我们能够通过策略改进定理知道策略的好坏，那么接下来介绍如何进行策略改进得到更多的回报。自然地，我们在每一个状态$s$下选择最优的$q_{\pi}(s,a)$即可，即新的策略$\pi'$满足：")
    st.latex(r'''\begin{aligned} \pi'(s)&≐\argmax_{a} q_{\pi}(s,a) \\
        &=\argmax_{a} \mathbb E[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_t=s,A_t=a] \\
            &=\argmax_{a} \sum_{s',r}p(s',r|s,a)(r+\gamma v_{\pi}(s'))\end{aligned} \tag{10.10}''')
    st.markdown("&emsp;&emsp;根据这种贪心策略改进的策略一定不亚于原策略，倘若改进后得到的新策略和原策略一样好，那么一定有$v_{\pi}$=$v_{\pi'}$，且均为最优策略，然我们来看看为什么：")
    st.latex(r''' \begin{aligned} v_{\pi}(s)=v_{\pi'}(s)&≐\argmax_{a}q_{\pi}(s,a) \\ 
             &=\argmax_a q_{\pi}(s,\pi'(s)) \\
                 &=\argmax_a \sum_{s',r} p(r,s'|s,a)[r+\gamma v_{\pi}(s')]\end{aligned} \tag{10.11}''')
    st.markdown("&emsp;&emsp;式$10.11$和式$10.7$贝尔曼最优方程一样，所以说$v_{\pi}=v_{\pi'}$时原策略$\pi$和贪心策略改进得到的新策略$\pi'$就是最优的策略。")
    
    
    st.markdown("#### :blue[10.3.3 策略迭代]")
    st.markdown("#### :blue[10.3.4 价值迭代] ")
    
    
def reinforcement_montecarlo():
    st.markdown(":blue[学习中...........................]")
    pass

def reinforcement_gradient_policy():
    st.title("策略梯度方法")
    st.markdown("&emsp;&emsp;在强化学习中，。。。。")
    st.markdown("## :blue[策略近似]")
    
def reinforcement_TimeDifference():
    st.title("时序差分算法")
    
    pass
def reinforcement_ppo_family():
    st.title("PPO")
    st.markdown("&emsp;&emsp;PPO 是一种属于策略梯度算法的改进方法，旨在解决传统策略梯度算法中的一些问题，例如样本效率和迭代稳定性。PPO 提出了一种基于重要性采样比例和截断优化的策略更新方式，以提高采样数据的利用效率，并通过一定的限制确保策略更新的稳定性。")

pages={
    "强化学习基础": reinforcement_foundation,
    "蒙特卡洛方法": reinforcement_montecarlo,
    "时序差分方法": reinforcement_TimeDifference,
    "策略梯度方法": reinforcement_gradient_policy,
    "Proximal Policy Optimization": reinforcement_ppo_family
}
# 添加侧边栏菜单
selection = st.sidebar.radio("学习列表", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()