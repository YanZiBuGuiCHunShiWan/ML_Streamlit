import streamlit as st

def Agents_prompt_engineering():
    st.markdown("## :blue[Prompt Engineering]")
    #st.markdown("## :blue[[Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering)]")
    st.markdown("#### :blue[Basic Prompting]")
    st.markdown("&emsp;&emsp;Zero shot 和 Few shot是提示工程中最基础的两个方法，通常也被用于测评大语言模型的性能。")
    st.markdown("#### :blue[Zero-shot]")
    st.markdown("&emsp;&emsp;Zero-shot就是指直接输入问题，然后让模型输出结果，比如：")
    zero_shot_code="Text: i'll bet the video game is a lot more fun than the film.\nSentiment:"
    st.code(zero_shot_code,language="shell")
    st.markdown("#### :blue[Few-shot]")
    st.markdown("&emsp;&emsp;Few-shot可以认为是先给了几个案例，然后让模型参考案例后来预测结果；这种方式可以使模型更好地理解人类的意图和想要什么样的答案。比如：")
    few_shot_code="Text: (lawrence bounces) all over the stage, dancing, running, sweating, mopping his face and generally displaying the wacky talent that brought him fame in the first place. \nSentiment: positive \n\n\
Text: despite all evidence to the contrary, this clunker has somehow managed to pose as an actual feature movie, the kind that charges full admission and gets hyped on tv and purports to amuse small children and ostensible adults. \nSentiment: negative \n\n\
Text: for the first time in years, de niro digs deep emotionally, perhaps because he's been stirred by the powerful work of his co-stars.\nSentiment: positive\n\n\
Text: i'll bet the video game is a lot more fun than the film.\nSentiment:"
    st.code(few_shot_code,language="markdown")
    st.markdown("#### :blue[Instruction-prompting]")
    st.markdown("&emsp;&emsp;Few-shot方式是给模型展示了我们的意图，同时也拓展了输入文本的长度，增大了计算耗时。那么为什么不直接给出具体的指令呢？Instruct GPT用高质量的指令数据微调\
        了大语言模型，并使用人类反馈强化学习让模型输出的内容更加符号人类价值观，大大降低了和大语言模型的'沟通成本'。")
    st.markdown("&emsp;&emsp;在和指令微调后的大语言模型对话时，我们应该详细地描述任务：")
    instruct_code='''Please label the sentiment towards the movie of the given movie review. The sentiment label should be "positive" or "negative". \nText: i'll bet the video game is a lot more fun than the film. \nSentiment:'''
    st.code(instruct_code,language="markdown")
    st.markdown("&emsp;&emsp;Incontext-learning([Ye et al.2023](https://arxiv.org/abs/2302.14691))则是将Few-shot和Instruction-prompting进行了结合，它在提示中包含了不同任务的多个示例，每个示例都由指令、任务输入和输出组成。但是他们的实验仅针对分类任务，且在instruction prompt中声明了所有的类别标签。示例如下：")
    incontext_learning_code='''Definition: Determine the speaker of the dialogue, "agent" or "customer".
Input: I have successfully booked your tickets.
Ouput: agent

Definition: Determine which category the question asks for, "Quantity" or "Location".
Input: What's the oldest building in US?
Ouput: Location

Definition: Classify the sentiment of the given movie review, "positive" or "negative".
Input: i'll bet the video game is a lot more fun than the film.
Output:'''
    st.code(incontext_learning_code,language="markdown")
    st.markdown("#### :blue[Chain-of-Thought (CoT)]")
    st.markdown("&emsp;&emsp;思维链（CoT）可以通过逐步描述推理逻辑，一步步地得出最终答案。其好处对于复杂的推理任务较为明显（大型模型参数超过50B）。简单的任务只会从CoT提示中稍微受益。")
    st.markdown("##### 两种类型的CoT")
    st.markdown("- Few shot CoT：给模型提供几个示例，每一个示例包含了高质量的推理链，案例如下：")
    Few_shot_cot_code='''Question: Tom and Elizabeth have a competition to climb a hill. Elizabeth takes 30 minutes to climb the hill. Tom takes four times as long as Elizabeth does to climb the hill. How many hours does it take Tom to climb up the hill?
Answer: It takes Tom 30*4 = <<30*4=120>>120 minutes to climb the hill.
It takes Tom 120/60 = <<120/60=2>>2 hours to climb the hill.
So the answer is 2.
===
Question: Jack is a soccer player. He needs to buy two pairs of socks and a pair of soccer shoes. Each pair of socks cost $9.50, and the shoes cost $92. Jack has $40. How much more money does Jack need?
Answer: The total cost of two pairs of socks is $9.50 x 2 = $<<9.5*2=19>>19.
The total cost of the socks and the shoes is $19 + $92 = $<<19+92=111>>111.
Jack need $111 - $40 = $<<111-40=71>>71 more.
So the answer is 71.
===
Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?
Answer:
'''
    st.code(Few_shot_cot_code,language="markdown")
    st.markdown("- Zero-shot CoT：使用自然语言语句，如“让我们一步一步地思考”，明确鼓励模型首先生成推理链，然后提示。案例如下：")
    Zero_shot_cot_code='''Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?
Answer: Let's think step by step.
'''
    st.code(Zero_shot_cot_code,language="markdown")
    st.markdown("#### :blue[Automatic Prompt Design]")
    st.markdown("&emsp;&emsp;Prompt其实就是指一系列前缀标记，增加了在给定输入下模型成功预测真实标签的概率，如AutoPrompt([Shin et al.(2020)](https://arxiv.org/abs/2010.15980)),Prefix Tuning([Li&Liang(2021)](https://arxiv.org/abs/2101.00190)),P-Tuning([Liu et al.2021](https://arxiv.org/abs/2103.10385))和Prompt Tuning([Lester et al.2021](https://arxiv.org/abs/2104.08691))\
        将这些前缀视作可训练的参数，通过梯度下降直接在嵌入空间对参数进行优化。")
    st.title("Smart Prompt Design")
    st.markdown("&emsp;&emsp;大语言模型已被证实，即使没有经过微调，只设计提示就能在许多NLP任务上有优越的性能。但是设计提示对下游任务的结果影响比较显著，且通常耗时耗力。手动设计提示的工作似乎就像\
        统计机器学习学习中的特征工程，绞尽脑汁地清洗、构造特征。")
    st.markdown("### :blue[Gradient-based Search]")
    st.markdown("&emsp;&emsp;AutoPrompt([Shin et al.,2020](https://arxiv.org/abs/2010.15980))就是基于梯度搜索的自动创建各种任务提示的方法。\
        AutoPrompt方法构造了一系列的Trigger Token插入到输入和带预测标签之间，对于所有有输入是共享的，因此普遍有效。")
    st.image("src/autoprompt.png")
    st.markdown("&emsp;&emsp;对于数据集中的所有输入而言，Trigger Token都可以用来优化目标：")
    st.latex(r'''x_{\text {trig }}=\arg \min _{x_{\text {trig }}^{\prime}} \mathbb{E}_{x \sim \mathcal{X}}\left[\mathcal{L}\left(\tilde{y}, f\left(x_{\text {trig }}^{\prime} ; x\right)\right)\right]''')
    st.markdown("&emsp;&emsp;梯度搜索的空间只局限于embedding空间，每一个Trigger Token的embedding,记作${e_{\\text {trig }_{i}}}$都会被初始化为默认值，然后遵循如下公式更新Trigger Token的embedding参数：")
    st.latex(r'''e_{\text {trig }}^{(t+1)}=\arg \min _{e \in \mathcal{V}}\left[e-e_{\text {trig }_{i}}^{(t)}\right]^{\top} \nabla_{e_{\text {trig }_{i}}^{(t)}} \mathcal{L}''')
    st.markdown("&emsp;&emsp;其中，$\mathcal {V}$代表所有tokens的embedding矩阵。$\\nabla {e_{\\text {trig }_{i}}^{(t)}} \mathcal {L}$则是当前batch的任务损失对应的梯度均值，因此上述的计算几乎要遍历所有的embedding。\
        我们可以通过$\mathcal V$维点积来暴力求解最优解。（矩阵乘法可以并行计算）如下图所示：")
    st.image("src/universal-adv-triggers.png",caption="Fig.2.")
    st.markdown("&emsp;&emsp;Smart prompt design本质上就是产生了高效的上下文（语言模型能理解）可以充分挖掘大语言模型蕴含的语义知识来完成任务。受Auto Prompt的启发\
        Li&Liang(2021)提出了[Prefix-tuning](https://arxiv.org/abs/2101.00190)，该方法在输入序列的开头增加少量的可训练的参数。可训练的前缀参数$\mathcal {P_{\\theta}} \in R^{|\mathcal {P_{idx}} \\times dim({h_i})|}$，$h_i$形式如下：")
    st.latex(r'''h_{i}=\left\{\begin{array}{ll}
    P_{\theta}[i,:], & \text { if } i \in \mathcal{P}_{\text {idx }} \\
    \operatorname{LM}_{\phi}\left(z_{i}, h_{<i}\right), & \text { otherwise }
    \end{array}\right.''')
    st.markdown("&emsp;&emsp;在训练期间，只有参数$\mathcal {P_{\\theta}}$是可训练的，语言模型的参数是冻结的。")
    st.image("src/prefix-tuning.png",caption="Fig.3.")
    st.markdown("&emsp;&emsp;前缀参数和词嵌入并没有关联，理论上来讲表达能力更加丰富，但是直接优化参数$\mathcal {P_{\\theta}}$的效果并不好，论文中提到，为了减少训练高纬参数的难度，将$\mathcal {P_{\\theta}}$重参数化，分解为一个小的矩阵和一个前馈网络。")
    st.markdown("&emsp;&emsp;微调模型在数据量充分时可以获得更好的性能，但是低资源场景下效果可能不理想。AutoPrompt和P-tuning都证实了在数据量小时能够取得比微调更有效的结果，\
    设计prompt或者context learning相较于微调来说更加省资源。在NLI任务上AutoPrompt的准确率比Linear Probing更高，也比手动设置的提示模板更准确地检索事实。在低数据场景下，Prefix-tuning在table-to-text generation和summarization任务上的性能和微调相当。")
    st.markdown("#### :blue[P-tuning]")
    st.markdown("&emsp;&emsp;[P-tuning](https://arxiv.org/abs/2103.10385)也遵循了训练嵌入空间的思想，但是在可训练参数和模型架构上和之前的工作有些不同。")
    st.image("src/p-tuning.png",caption="Fig.4.")
    st.markdown("&emsp;&emsp;$[P_i]$记作提示模板中的第$i$个token，prompt则可以记作序列$T=\{[P_{0:i}],\mathbf x [P_{i+1:m}],\mathbf y\}$，其中$P_i$并不是一个实际的token，它是一个伪token，即不存在于此词表。\
        由Prompt Encoder对token编码后得到$T^{e}$，公式如下：")
    st.latex(r'''T^{e}=\left\{h_{0}, \ldots, h_{i}, \operatorname{embed}(\mathbf{x}), h_{i+1}, \ldots, h_{m}, \operatorname{embed}(\mathbf{y})\right\}''')
    st.markdown("&emsp;&emsp;$P_i$的参数通过梯度下降的方式进行优化。P-tuning主要有两个难点需要解决：")
    st.markdown("&emsp;&emsp;1.:red[离散性]：预训练语言模型的单词嵌入是高度离散的。如果随机初始化则很难进行优化。")
    st.markdown("&emsp;&emsp;2.:red[关联性]：$h_i$和其他的token有潜在的关联，因此作者引入了一个轻量级的$Lstm$来捕捉token间的关系：")
    st.latex(r'''h_{i}=\operatorname{MLP}\left(\left[\operatorname{LSTM}\left(h_{0: i}\right): \operatorname{LSTM}\left(h_{i: m}\right)\right]\right)''')
    st.markdown("#### :blue[Prompt tuning]")
    st.markdown("&emsp;&emsp;Prompt tuning简化了P-tuning的思想，只在下游任务的输入文本前加上k个参数可以调整的token，给定输入下模型生成标签的概率记作$p_{\\theta, \\theta_{P}}(Y\mid[P;X])$，\
        $P$就是参数为$\\theta_{P}的$pseudo prompt，可以通过反向传播学习。$X \in \mathbb R^{n \\times d^e}$和$P \in \mathbb R^{k \\times d^e}$都表示嵌入向量，")
    st.markdown("&emsp;&emsp;随着模型参数量的增加，Prompt tuning获得了和微调旗鼓相当的效果。通过学习特定领域的参数，Prompt tuning能够比微调更好地适应于新的领域。 作者也表明未来会引入Prompt 集成来提示效果。")
    st.image("src/prompt-tuning.png",caption="Fig.5.")
    
def Domain_specialization_as_the_key():
    st.markdown("## :blue[Domain Specialization as Key to Make Large Language Models Disruptive: A Comprehensive Survey]")
    st.markdown("&emsp;&emsp;自然语言处理（NLP）和人工智能（AI）模型的演进经历了一个引人注目的过程。它始于20世纪50年代和60年代的基于规则的系统，随后转向了20世纪90年代的统计模型，再到2010年代的神经网络。在这个演进的过程中，自注意力和Transformer架构的成功[152]催生了预训练语言模型（PLMs），并在2010年代末迅速崭露头角。\
                PLMs能够以无监督的方式从大规模数据中学习通用语言表示，这对于诸如常识推理、多项选择题回答和故事生成等许多下游NLP任务都具有潜在优势，并避免了需要从头开始训练新模型。")
    st.markdown("&emsp;&emsp;近年来，随着大规模语料库和硬件容量的迅速增长，研究人员发现通过扩大模型和训练数据的规模可以持续提高模型容量，遵循缩放定律。这最终催生了大语言模型（LLMs），如GPT-3（1750亿参数）、PaLM（5400亿参数）和LLaMA（650亿参数）。\
        与较小模型相比，LLMs在理解和生成类似人类文本方面表现显著，成为令人期待的人工智能研究趋势。它们有望通过高效的文献分析、新假设生成和复杂数据解释，彻底改变自然和社会科学的研究方法，加速发现过程，并促进跨学科合作。")
    st.markdown("&emsp;&emsp;虽然大语言模型（LLMs）在作为通用任务解决器方面具有巨大的潜力，但要有效地将他们打造成一个合格的“聊天机器人”依然面临着重大挑战。这就导致了“LLMs领域专业化”的出现。具体而言，大语言模型的领域专业化被定义为根据特定领域的上下文数据进行定制的过程，通过领域特定的知识进行增强，通过领域目标进行优化，并受到领域特定约束的调控。对LLMs进行领域专业化的这种转变是出于几个引人注目的原因。\
        首先，不同领域、角色和任务之间的对话和语言风格存在显著差异，从医学处方到法律文件再到在线聊天等。获得这些能力和经验甚至需要人类多年的培训，其中许多是实践和专有的。此外，不同领域、机构和团队面对不同的业务有着自己的“商业模型”，单一通用的LLMs没有办法直接有效地解决特定领域下的问题。\
            其次，专业级使用对领域知识的要求也需要非常深入、实时和准确，而这些都不是通用的LLM所能轻易实现的。许多领域知识资源是机构、公司的专有资产和核心竞争力，绝不能泄露给通用LLM。")
    st.markdown("&emsp;&emsp;llm的领域化是一个具有挑战性的问题，其主要面临三个重大挑战。")
    st.markdown("- :red[1.知识的时效性]")
    st.markdown("- :red[2.对于一个LLM而言难以学习各领域的专业知识]")
    st.markdown("- :red[3.用于下游任务的计算资源]")
    
def Imitating_proerties_llm():
    st.markdown("## :blue[The False Promise of Imitating Proprietary LLMs]")
    
    pass

def ChainofThoughtPrompting():
    
    st.markdown("# :blue[$\\text{Chain of Thought Prompting}$]")
    st.markdown("### :blue[简介]")
    st.markdown("&emsp;&emsp;近年来，随着大规模语料库和硬件容量的迅速增长，研究人员发现通过扩大模型和训练数据的规模可以持续提高模型容量，这最终催生了大语言模型，如GPT-3（1750亿参数）、PaLM（5400亿参数）和LLaMA（650亿参数）。 \
        与较小模型相比，大语言模型在理解和生成文本方面表现显著，并逐渐成为热门的人工智能研究趋势。尽管大语言模型具有强大的语言生成能力，但是在推理类任务上（算术或者常识推理）表现仍有很大的进步空间，研究人员尝试了\
            在包含了不同推理问题的监督数据集上通过微调LLM的方式或者设计特定任务模块来解决此类问题，然而，:red[目前关于用few-shot learning方式来简化推理类任务的方法比较少。]")
    st.markdown("&emsp;&emsp;而:red[思维链(CoT)]就是最近提出的一种技术，它通过few-shot learning提升了LLM在推理类任务上的性能。CoT prompt在提示中插入几个推理问题的解决方案，然后每一个例子都和一个思维链相对应，LLM通过这些输入解决推理类问题。\
        这种方式的好处是只需要极少的数据（几个案例）变能显著提高LLM在推理任务上的性能，避免了极其耗用资源的微调方法。")
    st.markdown("### :blue[Prompting 和 Few-Shot Learning]")
    st.markdown("&emsp;&emsp;为了更好地理解CoT prompting，我们再次回顾一下Prompting和Few-Shot Learning。\
        在语言模型GPT和GPT2等提出以后，我们知道通过Next Token Prediction的方式进行预训练出来的模型非常强大，但是如何基于这些预训练模型来\
            解决特定领域的下游任务的方法并不明确。例如，我们可以用GPT在下游任务上微调模型，也可以基于GPT2用零样本(Zero-Shot)的方式解决问题。")
    st.image("src/AI-Agents/CoT-1.jpg",caption="Fig.1")
    st.markdown("&emsp;&emsp;在GPT-3提出以后，我们发现规模足够大的LLM通过Few-Shot Learning就可以表现得很好。而GPT-3参数有1750亿，可以在不经过微调的情况下\
        解决各种各样的任务，因此我们可以之间通过prompting的方式解决问题。")
    st.markdown("&emsp;&emsp;所谓prompting就是指通过输入如下文本的形式挖掘语言模型的知识。（prompting是一个过程，prompt就是提供给语言模型的输入。）")
    st.markdown("- $\\text{讲如下句子翻译为英文:<sentence> =>}$")
    st.markdown("- $\\text{根据给定的材料，阅读后进行总结:<sentence> =>}$")
    st.markdown("&emsp;&emsp;而我们可以编写合适的prompt来进行Zero-Shot Learning或者Few-Shot Learning。Zero-Shot Learning就是指输入给模型的prompt中，\
        没有提供正确的参考示例，让模型生成答案。Few-Shot Learning则是指输入的prompt中有少量的参考案例，其中，One-Shot Learning特指只有一个案例，下图是一个示例：")
    st.image("src/AI-Agents/CoT-2.jpg",caption="Fig.2")
    st.markdown("&emsp;&emsp;值得注意的是，大语言模型对prompt比较敏感，比如prompt中相邻两句话调换顺序或者删除几个字都可能显著影响大语言模型性能。")
    st.markdown("### :blue[大语言模型解决推理类任务]")
    st.markdown("&emsp;&emsp;既然参数够大的大语言模型蕴含了极其丰富的知识并可以很好地解决普通的任务，那么:blue[LLM在推理类的数据集上表现也会很好吗？我们扩大LLM的规模会提示其在推理类任务上的表现吗？]")
    st.markdown("- :red[Scaling up model size alone has not proved sufficient for achieving high performance on challenging tasks such as arithmetic, commonsense, and symbolic reasoning]-来自[[1]](https://arxiv.org/abs/2201.11903)")
    st.markdown("&emsp;&emsp;遗憾的是，更大规模的模型和预训练数据集无法提高LLM的推理能力。许多研究人员声称LLM只是在重复训练数据，而不是进行任何复杂地推理或者分析。")
    st.image("src/AI-Agents/CoT-3.jpg",caption="Fig.3:验证模块")
    st.markdown("&emsp;&emsp;:blue[**之前的方法：**]我们回顾一下在此前有哪些方法可用于解决推理类任务，:red[特定任务微调(Task-specific fine-tuning)]是解决算数、常识、符号推理任务的基本方法。进一步地，最好的方式是\
        训练一个额外的验证模块，该模块可以判断LLM生成的结果是否正确。在测试时，验证模块可以在LLM生成的几个答案上判断哪一个最好。如上图3所示。")
    st.markdown("&emsp;&emsp;虽然这些方法在某些场景下效果比较好，但是也存在一些局限性，比如：1.微调需要耗费计算资源。2.模型结构要适用于任务。3.大量的有监督微调数据集。\
        而通过promopting的方式去解决无疑更简单。")
    st.markdown("### :blue[**CoT prompting是如何起效的？**]")
    st.markdown("&emsp;&emsp;CoT说白了就是将思维链（一系列的推理步骤）注入到prompt中，再喂给大语言模型生成答案，更直白点就是在prompt中\
        写明了推理步骤给大语言模型参考。对于足够大（>100B）的大语言模型，这种方式可以显著提升LLMs在算数、常识和符号推理上的性能。")
    st.markdown("&emsp;&emsp;其实当人类在解决推理类任务时，将整体拆解为一个个子问题再逐一攻克是很常见的做法。比如我想\
        完成一个算数问题：:blue[我今年10岁，我表弟比我小3岁，10年以后表弟几岁？]")
    st.markdown("-  $\\text{今年我的岁数：10岁}$")
    st.markdown("-  $\\text{我表弟比我小3岁，表弟今年的岁数：10-3=7岁}$")
    st.markdown("-  $\\text{10年后表弟的岁数：7+10=17岁}$")
    st.markdown("&emsp;&emsp;虽然这个例子比较简单，但是这个想法可以拓展到人类是如何解决复杂的推理任务的。我们\
        通过生成一个思维链(在:blue[[[1]](https://arxiv.org/abs/2201.11903)]中定义为“一系列连贯的中间推理步骤”)得出最终的答案。\
            下图4是一些关于用思维链解决不同任务的例子。")
    st.image("src/AI-Agents/CoT-4.jpg",caption="Fig.4:思维链案例")
    st.markdown("&emsp;&emsp;CoT prompting在数学推理上的实验结果（来自于:blue[[1]]）如下图5所示：")
    st.image("src/AI-Agents/CoT-5.jpg",caption="Fig.5:实验结果")
    st.markdown("&emsp;&emsp;从图5的最左边中我们可以看到，当参数量比较小时（小于8B），参数量的增加不会给结果带来显著提升，当模型参数量\
        从小于10B到100B以上扩大的过程中，模型性能有了显著的飞跃。因此CoT prompting适用于更大的大语言模型，较小的模型会产生不符合逻辑的思维链。\
            从而降低性能。")
    st.markdown("&emsp;&emsp;:blue[[1]]中作者也对CoT prompting的稳健性进行了分析，比如改变prompt中例子的顺序等对最终结果的影响。结果表明，CoT prompting\
        对打乱案例顺序并不是特别敏感。当然，还有更多的细节，比如常识推理、符号推理任务上的性能在:blue[[1]]中都有说明，感兴趣的读者可以阅读原论文中相关片段。")
    
    st.markdown("### :blue[CoT Prompting的变种]")
    st.markdown("&emsp;&emsp;在CoT prompting提出以后，也出现了一些相关的变种，一些新奇的变种如下图[[2]](https://arxiv.org/abs/2205.11916)所示：")
    st.image("src/AI-Agents/CoT-6.jpg",caption="Fig.6:CoT变种示例")
    st.markdown("&emsp;&emsp;在Zero-shot-CoT中，没有提供详细的推理步骤，而是提供了一句话'Let's think step by step'。")
    st.markdown("&emsp;&emsp;:blue[**Self-consistency.**]Self-consistency 就是先用LLM通过采样的方法生成多个思维链，然后采用投票的方式选择最终的答案。如下图7所示：")
    st.image("src/AI-Agents/CoT-7.jpg",caption="Fig.7:Self-consistency")
    
    st.markdown("### :blue[总结]")
    st.markdown("&emsp;&emsp;在本文中，我们介绍了普通的prompt+LLM并不适用于推理类任务，而通过微调的方式解决推理类任务需要\
        计算资源和充分的下游任务数据，比较复杂。而通过CoT prompting即给定推理类任务的多个中间步骤注入到prompt中再输入给大语言模型\
            就可以取得良好结果，因此CoT是一种方便、简单的技术。此外，并不是所有规模的大语言模型都适用CoT，实验结果表明只有那些规模较大（>100B）的大语言模型\
                才能充分受益。最终，我们介绍了一些CoT的变种，对prompt的工作有了更加全面的了解。")
    st.markdown("### :blue[参考]")
    st.markdown('- [Wei, Jason, et al. "Chain of thought prompting elicits reasoning in large language models." arXiv preprint arXiv:2201.11903 (2022).](https://arxiv.org/abs/2201.11903)')
    st.markdown('- [Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.](https://arxiv.org/abs/2205.11916)')
    st.markdown('- [Chain of Thought Prompting for LLMs](https://cameronrwolfe.substack.com/p/chain-of-thought-prompting-for-llms)')
    
def LLM_Powered_Autonomous_Agents():
    st.markdown("## :blue[$\\text{LLM Agents}$]")
    st.markdown("&emsp;&emsp;随着大语言模型的兴起，$\\text{AI Agent}$这一词也随之变得火热，实际上$\\text{AI Agent}$已经经历了几个阶段的演变。我们可以先进行一下简短的回顾： ")
    st.markdown("&emsp;&emsp;:blue[$\\text{Symbolic Agents:}$]在人工智能研究早期阶段，$\\text{Symbolic AI}$占据了主导地位，其特点是依赖于符号性的逻辑。早期的$\\text{AI Agents}$主要关注两类问题：\
    :red[$\\text{1.transduction problem}$],:red[$\\text{2. respresentation/reasoning problem}$]。这些$\\text{Agents}$旨在模拟人类思考的方式，并且用于解决问题有明确的推理框架和可解释性。一个经典的例子就是专家系统，但是这类$\\text{Agents}$的缺陷也很明显，\
            符号主义在处理不确定性和大量现实世界问题时仍有局限性，此外，符号推理赛算法复杂性高，想要平衡其时间效率和性能具有十足挑战性。")
    st.markdown("&emsp;&emsp;:blue[$\\text{Reinforcement learning-based agents:}$]在强化学习（RL）的早期阶段，智能体主要依靠一些基础技术来进行学习，比如策略搜索和价值函数优化。其中，Q-learning和SARSA是比较著名的方法。\
        但是，随着深度学习技术的快速发展，我们将深度神经网络与强化学习相结合，形成了一种新的学习范式——深度强化学习（DRL）。这种结合让智能体能够从海量的高维数据中学习到复杂的策略，也带来了许多突破性的成果，比如AlphaGo和DQN。\
深度强化学习的强大之处在于，它让智能体能够在未知的环境中自主地进行学习，而不需要人类的干预或提供明确的指导。这种自主学习的能力，是AI领域的一大飞跃，也是未来智能体能够更好地适应复杂多变环境的关键。")
    st.markdown("&emsp;&emsp;:blue[$\\text{LLM-based agents:}$]随着大型语言模型（LLM）展现出令人瞩目的涌现能力，研究者们开始用其来打造新一代的人工智能代理。\
        他们将这些语言模型视为智能体的核心或“大脑”，并通过整合多模态感知和工具使用等策略，大大扩展了智能体的感知和行动能力。基于LLM的代理能够利用“思维链”（Chain of Thought，CoT）和问题分解（task decomposition）等技术，展现出与传统的符号智能体相媲美的推理和规划能力。\
            此外，它们还能够通过与环境的互动，从反馈中学习并执行新的动作。现如今，基于LLM的智能体已经被用于各种现实世界场景，比如用于软件开发与科学研究。:blue [[1](https://arxiv.org/abs/2309.07864v3)]")
    st.markdown("### :blue[$\\text{智能体系统概述}$]")
    st.markdown("&emsp;&emsp;在由LLM驱动的智能体系统中，三个关键的不同组件使得LLM能充当智能体的大脑，从而驱动智能体完成不同的目标。")
    st.markdown("- $\\text{Reasoning \& Planning.} $通过:blue[推理]，智能体可以将任务细分成更简单、可执行程度更高的子任务。就像人类解决问题一样，基于一些证据和逻辑进行推理，因此，对于智能体来说推理能力\
        对于解决复杂任务至关重要。:blue[规划]是人类应对复杂挑战时的核心策略。它帮助人类组织思维、确立目标，并制定实现这些目标的途径。对于智能体而言，规划能力同样关键，而这一能力取决于推理，\
        通过推理，智能体能够将复杂的任务分解为更易管理的子任务，并为每个子任务制定恰当的计划。随着任务的推进，智能体还能通过自省（反思）来调整其计划，确保它们与真实世界的动态环境保持一致，从而实现自适应性和任务的成功执行。\
        :red[总的来说，推理和规划可以将复杂任务拆分成更易解决的子任务，同时通过反省之前的推理步骤，从错误中学习，并为未来的行动精炼策略，以提高最终结果的质量。]")
    st.markdown("- $\\text{Memory.} $“记忆”存储智能体过去的观察、想法和行动的序列。正如人脑依赖记忆系统来回顾性地利用先前的经验来制定策略和决策一样，智能体需要特定的内存机制来确保它们熟练地处理一系列连续任务。\
        当人类面对复杂的问题时，记忆机制有助于人们有效地重新审视和应用先前的策略。此外，记忆机制使人类能够通过借鉴过去的经验来适应陌生的环境。此外，记忆可以分为短期记忆和长期记忆，对于基于LLM的智能体而言，$\\text{In-context learning}$的内容可以视作LLM的短期记忆，\
            而长期记忆则是指给LLM提供一个向量知识库，智能体可以通过检索知识库获取其内部信息。")
    st.markdown("- $\\text{Tool Use.} $当人们感知周围环境时，大脑会整合信息，进行推理和决策。人们通过神经系统控制身体，以适应或创造性行动，如聊天、避开障碍物或生火。\
        如果智能体拥有类似大脑的结构，具备知识、记忆、推理、规划和概括能力，以及多模式感知能力，那么它也被期望能够以各种方式对周围环境做出反应。而基于LLM的智能体的动作模块负责接收来自大脑模块的动作指令，并执行与环境互动的动作。\
        LLM收到指令再输出文本是其固有的能力，因此我们后继主要讨论其工具使用能力，也就是所谓的$\\text{Tool Use}$。")
    st.image("src/AI-Agents/llm-agents/agent_framework.jpg",caption="Fig.1. 基于LLM的智能体系统框架")
    st.markdown("### :blue[$\\text{1.Reasoning \& Planning(推理和规划)}$]")
    st.markdown("#### :blue[$1.1 \\text{Reasoning}$]")
    st.markdown("&emsp;&emsp;:blue[$\\text{Chain of Thought}$.]思维链技术逐渐成为大语言模型解决复杂类任务的标准方法，其通过在提示中加入几个具体的推理步骤提升大语言模型解决问题的性能，此外，有很多$\\text{CoT}$的变种，如\
        $\\text{Zero Shot CoT}$，其通过在提示中插入$\\text{\"think step by step\"}$这样的一句话引导模型思考，推理出最终答案。")
    st.markdown("&emsp;&emsp;:blue[$\\text{Tree of Thoughts}$.]$\\text{ToT}$通过在每个步骤探索多种推理可能性来扩展$\\text{CoT}$。它首先将问题分解为多个思考步骤，并在每个步骤中生成多个想法，从而创建一个树状结构。\
       然后基于树状结构进行搜索寻求最优结果，搜索过程可以是$\\text{BFS}$（广度优先搜索）或$\\text{DFS}$（深度优先搜索），每个状态由分类器（通过提示）或多数投票评估。")
    st.markdown("#### :blue[$1.2 \\text{Planning}$]")
    st.markdown("&emsp;&emsp;:blue[$\\text{Least-to-Most}$]​是提出问题后，先将其分割成若干小问题，然后一一解决这些小问题的一种策略。这种策略受到真实教育场景中用于指导儿童的策略的启发。\
    与$\\text{CoT Prompting}$类似，这个策略首先将需要解决的问题分解成一系列子问题。子问题之间存在逻辑联系和渐进关系。在第二步，逐一解决这些子问题。\
    与$\\text{CoT Prompting}$的最大差异在于，在解决下一个子问题时，会将前面子问题的解决方案作为提示输入。一个具体的字母连接的例子如下：")
    least2most_prompt='''
    Q: think, machine
    A: The last letter of "think" is "k". The last letter of "machine" is "e". 
    Concatenating "k" and "e" gives "ke". So "think, machine" output "ke".

    Q: think, machine, learning
    A: "think, machine" outputs "ke". The last letter of "learning" is "g". 
    Concatenating "ke" and "g" gives "keg". So "think, machine, learning" is "keg".

    Q: transformer, language
    A: The last letter of "transformer" is "r". The last letter of "language" is "e". 
    Concatenating "r" and "e" gives "re". So "transformer, language" is "re".

    Q: transformer, language, vision
    A: 
    '''
    st.code(least2most_prompt)
    st.image("src/AI-Agents/llm-agents/least2most.png",caption="Fig.3.Least-to-most 案例")
    st.markdown("&emsp;&emsp;:blue[$\\text{ReAct}$]通过将动作空间扩展为特定任务的离散动作和语言空间组合，成功将推理和行动集成在LLM中。前者能够使LLM生成自然语言进行推理，分解任务，后者则可以使LLM能够与环境交互（例如使用搜索API）。\
        $\\text{ReAct prompt}$提示模板包含让LLM思考的明确步骤，从$\\text{Langchain}$的源码中，我们可以找到如下：")
    ReAct_prompt="""
    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
    The only values that should be in the "action" field are: {tool_names}
    The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
    ```
    {{{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
    }}}}
    ```
    ALWAYS use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question"""
    st.code(ReAct_prompt)
    st.image("src/AI-Agents/llm-agents/ReAct.png",caption="Fig.2.ReAct Prompt推理轨迹")
    st.markdown("&emsp;&emsp;:blue[$\\text{Reflexion}$]是一个为智能体提供动态记忆和自我反思能力从而提升智能体推理能力的框架。如下图所示，$\\text{Reflexion}$框架中有三个角色，分别是:blue[$\\text{Actor,Evaluator,Self-Reflection}$]，\
        其中Actor模型基于观察到的环境和生成文本和动作，动作空间遵循$\\text{ReAct}$中的设置，在$\\text{ReAct}$中，特定任务的动作空间用语言扩展，以实现复杂的推理步骤。然后生成的这些文本和动作轨迹由$\\text{Evaluator}$进行评估，比如用0,1来表示好坏，接着$\\text{Self-Reflection}$\
        会生成特定的文本反馈，提供更丰富、有效的反思并被存储到记忆中。最后，$\\text{Actor}$会根据得到的记忆生成新的文本和动作直到完成任务。")
    st.image("src/AI-Agents/llm-agents/Reflexion.png",caption="Fig.4.Reflexion 算法框架")
    st.markdown("&emsp;&emsp;论文中提到，在序列决策任务$\\text{ALFWorld}$上，为了实现自动化评估采用了两个技术：1.用LLM进行二分类 2.人工写的启发式规则检测错误。对于后者，简单来说就是\
        如果智能化执行相同操作并收到相同响应超过3个周期，或者当前环境中的操作次数超过30次(计划不高效)，就进行自我反思。")
    st.markdown("&emsp;&emsp;:blue[$\\text{Chain of Hindsight(CoH).}$]是一种引入了人类偏好的训练方法，该方法不仅使用正面反馈数据，还利用了负面反馈数据，此外在模型预测时引入了反馈条件\
        ，使模型可以根据反馈学习并生成更符合人类偏好的内容。但$\\text{CoT}$利用了序列形式的反馈，在训练时给模型提供更丰富的信息，具体如下：")
    CoT_example='''How to explain a neural network to a 6-year-old kid? Bad:{a subpar answer} Good:{a good answer}.'''
    st.code(CoT_example)
    st.markdown("&emsp;&emsp;这样的数据拼接格式能组合不同种类的反馈，从而提升模型性能。在推理阶段，我们只需要在$\\text{prompt}$中给模型指定好$\\text{Good}$就能引导模型生成高质量的结果。\
        此外，在训练时并不是所有的$\\text{token}$都纳入损失函数计算，$\\text{feedback token}$即$\\text{Good or Bad}$只用来提示模型接下来预测时生成质量好的内容还是差的内容，具体损失函数公式如下：")
    st.latex(r'''\begin{aligned} \log p(\mathbf{x})=\log \prod_{i=1}^{n} \mathbb{1}_{O(x)}\left(x_{i}\right) p\left(x_{i} \mid\left[x_{j}\right]_{j=0}^{i-1}\right) \end{aligned}''')
    st.markdown("&emsp;&emsp;其中$\\mathbb{1}_{O(x)}$是指示函数，如果$x_i$属于$\\text{feedback token(Good or Bad)}$，则取值为$0$，反之为$1$。人类反馈数据，并不止单纯是先前所表述的只有$\\text{Good or Bad}$，它可以是更广义的形式，\
        我们可以记作$D_h=\{q,a_i,r_i,z_i\}^{n}_{i=1}$，$q,a_i$分别是问题和答案，$r_i$是答案的评级高低，$z_i$则是人类提供的事后反馈。假设反馈的评级由低到高排的顺序是$r_{n} \geq r_{n-1} \geq \cdots \geq r_{1}$,那么在微调时\
        数据拼接的形式就是$d_h=(q,z_i,a_i,z_j,a_j,\cdots,z_n,a_n),(i\leq j\leq n)$，训练时模型只会根据给定的前缀预测$a_n$，使得模型能够自我反映以基于反馈序列产生更好的输出。")
    st.markdown("&emsp;&emsp;在训练中，由于模型在预测时条件化了另一个模型输出及其反馈，因此可能会简单地“复制”已提供的示例，而不是真正理解任务。\
        为了防止模型只“复制”示例而不学习任务，作者在训练时随机掩盖了0%到5%的过去标记。这意味着模型无法直接看到这些标记，从而需要真正地理解上文，而不是简单地复制，这种随机掩盖增加了模型的泛化能力。")
    st.image("src/AI-Agents/llm-agents/CoT.png",caption="Fig.6.RLHF和CoT实验结果对比")
    st.markdown("&emsp;&emsp;该文对$\\text{RLHF}$和$\\text{CoT}$方法进行了实验对比，图中的蓝色柱是标准指令上的得分，从整体上看，$\\text{RLHF}$在标准指令上最终的效果还是要比$\\text{CoT}$要好的，但是当条件词改变，让模型输出更好的质量的内容时，$\\text{CoT}$生成的\
        结果质量明显要比$\\text{RLHF}$要好，而且对于不同条件词的变化敏感，说明模型较好地理解了人类偏好。")
    
    st.markdown("### :blue[$\\text{2.Memory(记忆)}$]")
    st.markdown("&emsp;&emsp;在神经科学领域，人类的记忆被分为几种不同的类型，这些类型在信息处理和存储方面各有特点，具体如下：")
    st.markdown("&emsp;:blue[$\\text{1.感觉记忆（Sensory Memory）}$]这是记忆系统的第一阶段，非常短暂，它保存了来自感官的信息，如视觉和听觉，通常只持续几秒钟。例如，当你看到一串数字时，你能在短时间内记住它们，这就像计算机中的缓存，用于临时存储。")
    st.markdown("&emsp;:blue[$\\text{2.短期记忆（Shot-term Memory）}$]短期记忆是你在短时间内能够主动保持和操纵的信息。它有有限的容量，通常可以持续大约20-30秒，通过重复或其他策略可以转移到长期记忆。这就像计算机的RAM，可以快速读写，但断电后信息会丢失。")
    st.markdown("&emsp;:blue[$\\text{3.长期记忆（Long-term Memory）}$]长期记忆是信息可以存储很长时间的地方，从几个小时到一生。它分为两种主要的子类型：")
    st.markdown("- :blue[$\\text{显性记忆（Explicit Memory）}$]这是有意识的记忆，可以进一步分为事实和信息（语义记忆）以及个人经历（情景记忆）。就像在硬盘上存储的文件，你可以有意识地检索这些信息。")
    st.markdown("- :blue[$\\text{隐性记忆（Implicit Memory）}$]这是无意识的记忆，包括技能、习惯和条件反射（骑单车、敲键盘）。这些记忆不像显性记忆那样容易被意识到，但它们影响你的行为和反应。这就像计算机的BIOS或操作系统设置，你通常不直接与它们交互，但它们影响计算机如何运行。")
    st.markdown("&emsp;&emsp;我们可以将智能体和记忆按照如下映射进行理解：")
    st.markdown("- 感觉记忆就像原始输入对应的语义嵌入，包括文本、图像和其他模态。")
    st.markdown("- 短期记忆就像上下文学习($\\text{In-context learning}$)，是有限的，受到$\\text{LLM}$上下文窗口限制。")
    st.markdown("- 长期记忆就像外部向量库，智能体可以通过快速检索从向量库中抽取相关的信息进行阅读理解。")
    st.markdown("&emsp;&emsp;外部记忆可以有效地缓解智能体幻觉问题，并能增强智能体在特定领域的性能。通常将相关非结构化知识由模型转化成语义嵌入并存储进向量数据库中，为了优化\
        检索速度，一般选择$\\text{ANN}$算法返回")
    st.markdown("### :blue[$\\text{3.Tool Use(工具使用)}$]")
    pass
