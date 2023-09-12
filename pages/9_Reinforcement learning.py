import streamlit as st

def reinforcement_foundation():
    st.sidebar.markdown("## 10.1 强化学习简介")
    st.sidebar.markdown("## 10.2 强化学习的最终目的")
    st.sidebar.markdown("### &emsp;10.2.1 策略和价值函数")
    st.sidebar.markdown("### &emsp;10.2.2 最优策略和最优价值函数")
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
    st.markdown("&emsp;&emsp;式子$10.3$和$10.3$被称作:red[贝尔曼期望方程]，其分别描述了:red[动作价值和后继动作价值之间的关系]以及:red[状态价值和后继状态价值之间的关系]。")
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
    
    
def reinforcement_dynamic_programming():
    st.sidebar.markdown("## 10.2 动态规划")
    st.sidebar.markdown("### &emsp;10.2.1 策略评估")
    st.sidebar.markdown("#### &emsp;&emsp;10.2.1.1迭代策略评估")
    st.sidebar.markdown("### &emsp;10.2.2 策略改进")
    st.sidebar.markdown("### &emsp;10.2.2 策略迭代")
    st.sidebar.markdown("### &emsp;10.2.3 价值迭代")
    st.sidebar.markdown("### &emsp;10.2.4 异步动态规划")
    st.sidebar.markdown("### &emsp;10.2.5 广义策略迭代")
    st.markdown("## :blue[10.2 动态规划]")
    st.markdown("### :blue[10.2.1 策略评估]")
    st.markdown("&emsp;&emsp;我们要计算出状态价值函数来才能进一步评估策略的好坏，因此接下来介绍如何计算出具体的状态价值函数。我们的条件是：给定$MDP$，:red[即我们知道\
        $p(s',r|s,a)$和策略$\pi$的条件下]求$v_{\pi}(s)$的表达式。由先前$10.3$可知：")
    st.latex(r'''\begin{aligned}v_{\pi}(s)&=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma v_{\pi}(s')] \\
        &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} rp(s',r|s,a)+\gamma \sum_{a \in \mathcal A}\pi(a|s) \sum_{s',r}p(s',r|s,a)v_{\pi}(s') \\
            &=\underbrace {\sum_{a \in \mathcal A}\pi(a|s)\sum_{r} rp(r|s,a)}_{\text{\textcircled a}}+\underbrace{\gamma \sum_{a \in \mathcal A}\pi(a|s)\sum_{s'} p(s'|s,a)v_{\pi}(s')}_{\text{\textcircled b}}\end{aligned}''')
    st.markdown("#### :blue[10.2.1.1 迭代策略评估]")
    st.markdown("&emsp;&emsp;解析解虽然能写出来，但是直接求解时间复杂度太高了，因此通常用数值解来求解最优策略，我们的目标是求出：")
    st.latex(r'''\mathbf v_{\pi}(s)=\begin{pmatrix} v_*(s_1)  \\ {\vdots} \\ v_*(s_{|\mathcal S|}) \end{pmatrix}''')
    st.markdown("&emsp;&emsp;在迭代的策略评估中，我们初始化一组序列$\{ v_k\}_{k=1}^{\infin}$，然后按照如下方程进行更新：")
    st.latex(r'''\begin{aligned}v_{k+1}(s)&≐\mathbb E_{\pi}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s] \\
        &=\sum_{a \in \mathcal A}\pi(a|s)\sum_{s',r} p(s',r|s,a) [r+\gamma v_{k}(s')]\end{aligned} \tag{10.9}''')
    st.markdown("&emsp;&emsp;最终一定会收敛到$v_{\pi}$。")
    st.markdown("### :blue[10.2.2 策略改进解]")
    
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
    
    st.markdown("### :blue[10.2.3 策略迭代]")
    st.markdown("&emsp;&emsp;所谓策略迭代，就是指迭代地找到最优策略$\pi_*$的过程，我们在前面两小节讲了策略评估和策略改进，实际上策略迭代就是这两个过程的结合。不断地交替两个过程从而找到最优策略。")
    st.latex(r'''\pi_0 \xrightarrow{Evaluate} v_{\pi_0} \xrightarrow{Improve} \pi_1 \xrightarrow{Evaluate} v_{\pi_1} \xrightarrow{Improve} \cdots \pi_* \xrightarrow{Evaluate} v_{\pi_*} ''')
    st.markdown("&emsp;&emsp;在有限马尔可夫决策过程中，整个定义空间都是有限的，策略也必然是有限的，在我们的交替过程中，值函数单调上升且显然有上界，当策略迭代过程执行一定步数后，策略必然能收敛到最优策略。策略迭代算法过程可见下图：")
    st.image("src/policy_iteration.png")
    st.markdown("&emsp;&emsp;可见在策略评估这一步要进行一个循环，然后到了策略改进又要进行一个循环，所以原始的策略迭代算法是比较耗时间的。;值得注意的是，价值迭代只是在策略评估这一过程中只走一步，并不代表不进行策略改进。")
    st.markdown("### :blue[10.2.4 价值迭代] ")
    st.markdown("&emsp;&emsp;实际上我们可以提前截断策略评估过程，并且不影响其收敛性，最极端的一种情况就是在对每个状态进行一次\
        更新后就停止策略评估，该算法称为:red[价值迭代]。")
    st.latex(r'''\begin{aligned} v_{k+1}(s)&≐\max_a\sum_{r,s'}p(s',r|r,s)(r+\gamma v_{k}(s')) \end{aligned} \tag{10.12}''')
    st.markdown("&emsp;&emsp;算法的具体过程如下图：实际上就是结合了策略评估和策略迭代，非常暴力。当然，:red[可以认为价值迭代是一种特殊的策略迭代。]")
    st.image("src/value_iteration.png")
    st.markdown("## :blue[10.3 异步动态规划]")
    st.markdown("&emsp;&emsp;在前述的迭代方法中，我们每走一步，都需要对所有的$v_{\pi}(s),s\in\mathcal S$进行更新，当状态空间非常大时，走一步的耗时也是非常高的。因此，我们可以考虑不去更新所有的状态，而是优先更新我们认为值得更新的状态，异步价值迭代的其中一个版本就是\
        利用截至迭代的更新公式$10.12$，在每一步的基础上只更新一个状态$s_k$。如果$0\leq\gamma\leq 1$，则只要所有状态都在序列$\{v_k\}$中出现无数次，就能保证渐近收敛到$v_*$。")
    st.markdown("## :blue[10.5 广义策略迭代]")
    st.image("src/GPI.png")
    st.markdown("&emsp;&emsp;广义策略迭代(Generalized Policy Iteration (GPI))就是描述策略评估和策略改进交替进行的过程，几乎所有的强化学习方法都能很好地被描述为 GPI 模型。\
       ")
    st.markdown("&emsp;&emsp; 广义策略迭代中的策略评估和策略改进的过程就如同'竞争和协作'的关系，评估过程旨在提供准确的信息和反馈，以便于制定改进策略的决策，策略改进的目标是寻找更好的策略来最大化性能。它可以采用不同的方法，如价值迭代、策略迭代或蒙特卡洛方法等，以搜索更优的行动或策略。改进过程会尝试探索其他可能的策略，并与当前策略进行比较，以确定哪些行动或策略更能提高性能，这是二者间'竞争性'的体现，\
        另一方面，评估过程提供了改进过程所需的关键信息和指导，帮助确定改进的方向。改进过程的结果又会被用于评估过程中验证和确定改进后的策略的优劣。通过循环迭代并相互协作，评估和改进过程共同促进了策略的优化。")
    
def reinforcement_montecarlo():
    st.markdown("## :blue[10.3 蒙特卡洛方法]")
    st.markdown("&emsp;&emsp;在前一章中，我们学习的是如何利用动态规划的方式求解出最终的策略$\pi$，这种方法有一个:red[前提就是动态特性$p(r,s'|s,a)$必须已知。]但是在绝大部分情况下，我们很难得到显示的分布，但是从希望的分布进行采样则较为容易。")
    st.markdown("&emsp;&emsp;我们知道状态价值函数是从该状态开始的期望回报，虽然我们不知道动态特性，但是如果有回报的观测值即经验，那么随着越来越多的回报被观测到，我们可以用回报的均值来近似状态价值。这就是蒙特卡洛方法的思想。")
    st.markdown("### :blue[10.2.1 状态价值的估计] ")
    st.markdown("- **首次访问型估计** ")
    st.markdown("&emsp;&emsp;假设给定策略$\pi$下途径状态$s$的多幕数据，这组多幕数据中，状态$s$可能会被多次访问到，我们称第一次访问为$s$的是$s$**的首次访问**。而首次访问型MC算法就是用状态$s$的所有首次访问的回报的平均值估计$v_{\pi}(s)$。\
        即$v_{\pi}(s) \\approx \\frac{1}{N} \\sum_{i=1}^N G_t^{(i)}$。")
    st.markdown("&emsp;&emsp;下图展示的是首次访问型MC估计状态价值的原理：")
    st.image("src/mc_value.jpg")
    st.markdown("&emsp;&emsp;伪代码如下图，我们只计算每一幕数据中状态$s$首次出现后的回报的期望，当这一幕数据中再次遇到状态$s$时，我们忽略。")
    st.image("src/mc_value_pscode.png")
    st.markdown("- **每次访问型估计** ")
    
    st.markdown("### :blue[10.2.2 动作价值的估计] ")
    st.markdown("&emsp;&emsp;和状态价值估计类似，动作价值的蒙特卡洛估计需要对“状态-动作”二元组$(s,a)$访问，首次访问型MC则会将每一幕首次访问到这个“状态-动作”\
        二元组得到的回报的平均值作为动作价值的近似。和先前一样，当对每个“动作-价值”二元组的访问次数趋向于无穷时，这些方法会收敛到动作价值函数的真实期望值。")
    st.markdown("&emsp;&emsp;虽然想法很好，但是我们不能忽略这样的一个可能性，即某一项“动作-状态”二元组是永远不会被访问到的，因为$\pi$是一个确定性策略时，那么遵循该确定性策略意味着\
        每一个状态只会观测到一个确定动作的回报，在无法获取其他动作的回报时蒙特卡洛算法无法根据经验改善动作价值函数的估计。")
    st.markdown("&emsp;&emsp;为实现动作价值函数的估计，我们有:blue[**试探性出发假设**]，即所有的“动作-状态”二元组都有非零的概率被选择为一幕数据的起点，这样可以保证\
                在采样的幕数趋于无穷时，每一个“动作-状态”二元组都会被访问到无数次。但是在真实环境下，制造满足该假设的条件可能根本没有办法做到，因此试探性出发假设只是理论上能做到。\
                    想要真正的估计动作价值函数，那么:blue[**无限幕数据**]和:blue[**满足试探性出发假设**]这两点是必须要克服的。")
    st.markdown("### :blue[10.2.3 蒙特卡洛控制]")
    st.markdown("&emsp;&emsp;:red[为了得到一个实际可应用的算法，我们必须想办法去除这两个假设]。我们先保留试探性假设出发来完成完整的蒙特卡洛控制(在这里将蒙特卡洛采样称作蒙特卡洛控制)。在进行策略评估时的无限多幕数据这一假设，实际上比较好去除。\
        和价值迭代的思想类似，我们不期望动作价值函数要经过很多步的迭代后接近真实值，而是每一幕结束后使用观测到的回报进行策略评估，然后在改幕序列访问到的每一个状态上进行策略改进。策略改进\
            就是上一章的贪心策略，即$\pi'(s)=\\argmax_{a} q_{\pi}(s,a)$。")
    st.markdown("### :blue[10.2.4 没有试探性出发假设的蒙特卡洛控制]")
    st.markdown("&emsp;&emsp;为了绕开难以满足的试探性出发假设，我们有两种方法：（1）同轨策略$(on-policy)$ （2）离轨策略$(off-policy)$。在同归策略中，:red[用于生成采样数据序列的策略\
        和用于实际决策的待评估与改进的策略是相同的]；在离轨策略中，:red[用于评估和改进的策略与生成采样数据的策略是不同的]。")
    st.markdown("&emsp;&emsp;在同轨策略中，策略一般都是软性的，即$\\forall s \in \mathcal S,a \in \mathcal A$有$\pi(a|s)\gt 0$。它们会渐渐地逼近确定性的策略，下面介绍的同轨策略方法\
        称为:blue[$\\varepsilon$-贪心策略]：")
    st.latex(r'''\pi^{\prime}(a \mid s)=\left\{\begin{array}{ll}
       1-\varepsilon+\frac{\varepsilon}{|\mathcal{A}(s)|} & , if \text { } a=a^*=\argmax_aq_{\pi}(s,a)\\
        \frac{\varepsilon}{|\mathcal{A}(s)|} & , \text { otherwise }    
        \end{array}\right. \tag{10.13}''')
    st.markdown("&emsp;&emsp;有了这个策略，我们当然需要比较这个策略和原有策略的好坏，和策略评估那一章一样，我们比较$q_{\pi}(s,\pi'(s))$和$v_{\pi}(s)$的大小。")
    st.latex(r'''\begin{aligned} q_{\pi}(s,\pi'(s))&=\sum_{a}\pi(a|s)q_{\pi}(s,a) \\
        &=\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)+(1-\varepsilon) \max _{a} q_{\pi}(s, a) \end{aligned} \tag{10.14}''')
    st.markdown("&emsp;&emsp;我们的目的是比较大小，而比较大小肯定涉及到缩放，而缩放则会自然地想到在$\max$这个符号项做文章。可以确定的是$\max_{a}q_{\pi}(s,a)\geq q_{\pi}(s,a)$。")
    st.latex(r'''\begin{aligned} 1=\frac{\sum_a \pi(a|s)-\frac{\varepsilon}{|\mathcal A(s)|}}{1-\varepsilon}  \end{aligned} ''')
    st.markdown("&emsp;&emsp;故有：")
    st.latex(r'''\begin{aligned}({1-\varepsilon})\max_{a}q_{\pi}(s,a)&\geq ({\sum_a \pi(a|s)-\frac{\varepsilon}{|\mathcal A(s)|}})q_{\pi}(s,a)\\
        &={\sum_a \pi(a|s)q_{\pi}(s,a)-\frac{\varepsilon}{|\mathcal A(s)|}}q_{\pi}(s,a)\end{aligned} ''')
    st.markdown("&emsp;&emsp;上述结果带入到$10.14$则有：")
    st.latex(r'''\begin{aligned} q_{\pi}(s,\pi'(s))&=\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)+(1-\varepsilon) \max _{a} q_{\pi}(s, a) \\
            &\geq \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) + {\sum_a \pi(a|s)q_{\pi}(s,a)-\frac{\varepsilon}{|\mathcal A(s)|}}q_{\pi}(s,a) \\
                &= \sum_a \pi(a|s)q_{\pi}(s,a) \\
                    &= v_{\pi}(s) \end{aligned} \tag{10.15}''')
    st.markdown("&emsp;&emsp;故有$\pi' \geq \pi$。")
    st.markdown("### :blue[10.2.5 基于重要度采样的离轨策略]")
    st.markdown("&emsp;&emsp;几乎所有的离轨策略方法都采用了重要度采样，对于一幕数据而言，其子序列的概率为：")
    st.image("src/RL-off-policy.png")
    st.markdown("&emsp;&emsp;给定起始状态$S_t$，后续:red[状态-动作]轨迹为$A_t,S_{t+1},...,S_T$在策略$\pi$下发生的概率是：")
    st.latex(r'''\begin{aligned} &Pr\{ A_t,S_{t+1},A_{t+1},...,S_T|S_t,A_{t:T-1}\sim \pi\} \\
        &= \pi(A_t|S_t)p(S_{t+1}|A_t,S_t)\pi(A_{t+1}|S_{t+1})......\pi(A_{T-1}|S_{T-1})p(S_T|S_{T-1},A_{T-1}) \\
            &=\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|A_k,S_k) \end{aligned} ''')
    st.markdown("&emsp;&emsp;同理，在策略$b$下动作-状态序列出现的概率为：")
    st.latex(r'''\begin{aligned} &Pr\{ A_t,S_{t+1},A_{t+1},...,S_T|S_t,A_{t:T-1}\sim b\} \\
        &= b(A_t|S_t)p(S_{t+1}|A_t,S_t)b(A_{t+1}|S_{t+1})......b(A_{T-1}|S_{T-1})p(S_T|S_{T-1},A_{T-1}) \\
            &=\prod_{k=t}^{T-1} b(A_k|S_k)p(S_{k+1}|A_k,S_k) \end{aligned}''')
    st.markdown("&emsp;&emsp;对应的目标策略和行动策略轨迹下的相对概率（重要度采用比）是：")
    st.latex(r'''\rho_{t:T-1}≐\frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|A_k,S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)p(S_{k+1}|A_k,S_k)} \
        = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)} ''')
    st.markdown("&emsp;&emsp;可以看到，和动态特性$p$有关的项都消掉了，所以重要度采样比只和策略与样本数据相关，与动态特性无关。有了前面的准备，那么我们可以推出：")
    st.latex(r'''\begin{aligned} v_{\pi}(s)&=\mathbb E_{\pi}[G_t|St=s] \\
        &=\mathbb E_b[\rho_{t:T-1}G_t|St=s]\end{aligned} ''')
    st.markdown("&emsp;&emsp;如果观察到了一批遵循策略$b$生成的多幕采样序列，那么我们就可以用其回报进行平均来估计$v_{\pi}(s)$。对于每次访问型方法，我们定义所有访问过状态$s$的时刻集合为$\mathcal T(s)$:blue[（对于首次访问型而言，$\mathcal T$只包含了首次访问到$s$的时刻）]。\
        将时刻$t$后的首次终止表示为$T(t)$，$G_t$表示在$t$之后到达$T(t)$时的回报值，那么$\{ G_t\}_{t \in \mathcal T(s)}$则是状态$s$对应的回报值，$\{ \\rho_{t:T(t)-1} \}$是相应的重要度采样比集合，那么$v_{\pi}(s)$\
        就可以用重要度采样比调整回报值进行平均即可：")
    st.latex(r'''V(s)≐\frac{\sum_{t \in \mathcal T(s)}\rho_{t:T(t)-1}G_t}{|\mathcal T(s)|} \tag{10.16}''')
    st.markdown("&emsp;&emsp;上述方法是简单加权平均，优点是：无偏估计，缺点是：方差较大、不稳定。另一种方法是加权重要度采样，其定义为：")
    st.latex(r'''V(s)≐\frac{\sum_{t \in \mathcal T(s)}\rho_{t:T(t)-1}G_t}{\sum_{t \in \mathcal T(s)}\rho_{t:T(t)-1}} \tag{10.17}''')
    st.markdown("&emsp;&emsp;该方法的缺点是：对$v_{\pi}(s)$是无偏估计，优点是：方差小，较稳定。")
    st.image("src/RL-variance.png")
    st.markdown("&emsp;&emsp;上图是一个带环序列轨迹的例子，图中只有一个非终止状态$s$和左右两种动作，采取向左的动作有0.1的概率转移到终止状态并得到收益1，有0.9的概率\
        转移回$s$状态且没有收益。向右的动作则必定转移到终止状态且没有收益。图中的曲线是基于普通重要度采样的首次访问型MC算法的10次独立运行的结果。我们可以通过简单计算来确认:red[经过重要度采样比加权的回报的方差是无穷的。]")
    st.latex(r'''\mathbf {Var[\rho_{t:T-1}G_t]}=\mathbb {E}[(\rho_{t:T-1}G_t)^2]-{E}[\rho_{t:T-1}G_t]^2 \tag{10.18}''')
    st.markdown("&emsp;&emsp;上式的右半部分是期望的平方，肯定是有界的。而左半部分平方的期望则需要证明是有界还是无界。")
    st.latex(r'''\begin{aligned} \mathbb {E}_b[\rho_{t:T-1}G_t ^2] &= \mathbb {E}_b[(\prod_{t=0}^{T-1}\frac{\pi(A_t|S_t)}{S(A_t|S_t)}G_0)^2] \\
        &=\frac{1}{2}\cdotp 0.1(\frac{1}{0.5})^2 \\
            &+\frac{1}{2}\cdotp 0.9\cdot \frac{1}{2}\cdot 0.1(\frac{1}{0.5}\cdotp \frac{1}{0.5})^2 \\
                &+\frac{1}{2}\cdotp 0.9\frac{1}{2}\cdotp 0.9\cdot \frac{1}{2}\cdot 0.1(\frac{1}{0.5}\cdotp  \frac{1}{0.5} \cdotp \frac{1}{0.5})^2 \\
                    &+ \cdots \\
                        &=\sum_{t=1}^{\infty}(\frac{1}{2})^{t}0.9^{t-1}0.1((\frac{1}{0.5})^t)^2 \\
                            &=\infty\end{aligned}''')
    st.markdown("### :blue[10.2.6 增量式实现]")
    st.markdown("&emsp;&emsp;可以通过增量计算来节省内存并且提高计算速度，假设我们已有汇报序列$G_1,G_2,....,G_{n-1}$，每一个回报都对应一个随即权重$W_i$，那么有：")
    st.latex(r'''\begin{aligned}V_{n}&≐\frac{\sum_{k=1}^{n-1} W_{k} G_{k}}{\sum_{k=1}^{n-1} W_{k}} \\
        V_{n+1}&≐\frac{\sum_{k=1}^{n-1} W_{k} G_{k}+W_{n}G_{n}}{\sum_{k=1}^{n} W_{k}} \\
            &=\frac{\sum_{k=1}^{n-1} W_{k} G_{k} \frac{\sum_{k=1}^{n-1} W_{k}}{\sum_{k=1}^{n-1} W_{k}} +W_{n}G_{n}}{\sum_{k=1}^{n} W_{k}} \\
                &=\frac{V_nC_{n-1}+W_{n}G_{n}}{C_n}\\
                    &=\frac{V_n(C_{n}-W_n)+W_{n}G_{n}}{C_n} \\
                        &=V_n+\frac{(G_n-V_n)W_n}{C_n}\end{aligned} \tag{10.19}''')
    st.markdown("&emsp;&emsp;离轨策略的策略评估算法流程如下图所示：")
    st.image("src/RL-off-vallue.png")
    st.markdown("&emsp;&emsp;当权重$W$不为0时才会循环，其中$C(S_t,A_t)$就相当于公式$10.19$中的$C_n$")
    st.markdown("### :blue[离轨策略的蒙特卡洛控制]")
    st.markdown("&emsp;&emsp;动作价值的估计算法流程如下图所示：")
    st.image("src/RL-off-action.png")
    st.markdown("&emsp;&emsp;算法最后一行的的采样比为$\\frac{1}{b(A_t|S_t)}$是因为c此时的$\pi$是一个确定性的策略。")
    
def reinforcement_gradient_policy():
    st.title("策略梯度方法")
    st.markdown("&emsp;&emsp;在强化学习中，。。。。")
    st.markdown("## :blue[策略近似]")
    
def reinforcement_TimeDifference():
    st.markdown("## :blue[10.4 时序差分算法]")
    st.markdown("### :blue[10.4.1 TD Prediction]")
    st.markdown("&emsp;&emsp;让我们先回顾一下每次访问型的蒙特卡洛方法的状态价值函数估计：")
    st.latex(r'''V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[G_{t}-V\left(S_{t}\right)\right]''')
    st.markdown("&emsp;&emsp;上式中，$G_t$是通过不断采样，即走了很多步以后算出来的值；蒙特卡洛方法必须等待整幕数据结束以后才会进行一次更新。")
    st.markdown("&emsp;&emsp;本章的时序差分算法$TD(0)$的状态价值更新公式为：")
    st.latex(r'''V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right]''')
    st.markdown("&emsp;&emsp;TD(0)的更新公式的右半部分括号内实际上是一种误差，衡量的是$S_t$估计值与$R_t+\\gamma V(S_{t+1})$之间的差异。这个数值称作$TD$误差。")
    st.latex(r'''\delta_{t} \doteq R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)''')
    
    st.markdown("### :blue[10.4.2 Advantages of TD Prediction Methods]")
    st.markdown("&emsp;&emsp;接下来看一个马尔可夫收益过程的例子（没有动作）：")
    st.image("src/RL-TD-adv.png")
    st.markdown("&emsp;&emsp;根据状态和收益转移图我们可以列出以下方程：（:red[我们用$P_i$来表示当我们处于状态 i 下，最终能到达右终止态的概率]）")
    st.latex(r'''\left\{\begin{aligned}
P_{a} & =\frac{1}{2} P_{b}+\frac{1}{2} \times 0 \\
P_{b} & =\frac{1}{2} P_{a}+\frac{1}{2} P_{c} \\
P_{c} & =\frac{1}{2} P_{b}+\frac{1}{2} P_{d} \\
P_{d} & =\frac{1}{2} P_{c}+\frac{1}{2} P_{e} \\
P_{e} & =\frac{1}{2} P_{d}+\frac{1}{2} \times 1
\end{aligned}\right. \tag{10.20}''')
    st.latex(r''' \Rightarrow P_{a}=\frac{1}{6}, P_{b}=\frac{2}{6}, P_{c}=\frac{3}{6}, P_{d}=\frac{4}{6}, P_{e}=\frac{5}{6}''')
    st.markdown("&emsp;&emsp;由于只有状态$E$转移到终止状态会获得收益$+1$，因此这些状态价值就等于对应的状态概率。")
    st.markdown("&emsp;&emsp;下图展示了时序差分方法和蒙特卡洛方法在该任务上的表现：")
    st.image("src/RL-TD-adv2.png")
    st.markdown("&emsp;&emsp;左子图显示的是$TD$算法分别运行$1、10、100$次后与真实值的比较。可见当运行$100$次以后预测效果就十分接近真实值了。右子图则对比了\
        时序差分和蒙特卡洛方法在不同$\\alpha$取值下的误差收敛情况，可以明显观察到，在该任务上，时序差分方法收俩速度显著快于蒙特卡洛方法，且误差更小。")
    st.markdown("&emsp;&emsp;$TD$方法则只需等到这一步结束，利用实时观测到的奖励值和现有估计值来进行更新。$TD$算法的优点如下：:blue[1.无须知道动态特性。2.蒙特卡洛方法在幕数很长时有较大的延时问题，TD方法能够解决这种问题，可以通过在线、实时的方式进行增量更新。]")
    st.markdown("### :blue[10.4.3 Optimality of TD(0)]")
    st.markdown("### :blue[10.4.4 SARSA: On-policy TD Control]")
    st.markdown("&emsp;&emsp;时序差分算法的动作价值估计时，对于同轨策略，我们遵循当前策略$\pi$生成的数据（状态$s$和动作$a$）来估计$q_{\pi}(s,a)$")
    st.image("src/RL-TD-SARSA1.png")
    st.markdown("### :blue[10.4.5 Q-Learning: On-policy TD Control]")
    st.markdown("### :blue[10.4.6 Expected SARSA]")
    pass
def reinforcement_ppo_family():
    st.title("PPO")
    st.markdown("&emsp;&emsp;PPO 是一种属于策略梯度算法的改进方法，旨在解决传统策略梯度算法中的一些问题，例如样本效率和迭代稳定性。PPO 提出了一种基于重要性采样比例和截断优化的策略更新方式，以提高采样数据的利用效率，并通过一定的限制确保策略更新的稳定性。")
    
def reinforcement_prev():
    st.title("强化学导论")
    
pages={
    "强化学习导论": reinforcement_prev,
    "强化学习基础": reinforcement_foundation,
    "动态规划": reinforcement_dynamic_programming,
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