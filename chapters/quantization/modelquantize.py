import streamlit as st


def ModelQuantizationComprehend():
    st.markdown("## :blue[模型量化]")
    st.markdown("&emsp;&emsp;大语言模型的出现为人们提供了新的自然语言处理任务解决思路和方法，其强大的语言理解能力可以轻易地解决\
        文本总结、文章生成、情感分析等任务。然而使用大语言模型并不是一件轻而易举的事情，我们需要高性能的显卡才能支持模型推理，得到答案。\
            想要推理14B级别的大语言模型，需要显存大约要26G，也就是1张消费级显卡4090 Ti都跑不起来，此外，\
                如果我们想将现代网络集成到具有严格的功率和计算要求的边缘设备中，那么降低神经网络推理的功率和延迟是关键。\
                    :blue[神经网络量化是实现这些节省的最有效的方法之一]。")
    st.markdown("#### :blue[什么是量化？]")
    st.markdown("&emsp;&emsp;模型量化相当于讲原来高精度的数值转化为了精度较低的定点数值，同时尽可能减少计算精度损失的方法。比如将模型权重($\\text{float32}$)转化完$\\text{int8}$类型。")
    st.markdown("&emsp;&emsp;在机器学习的背景下，浮点数据类型也被称为“精度”。模型的大小由其参数的数量及其精度决定，通常是$\\text{float32、float16}$或$\\text{bfloat16}$中的一个，如[下图](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)所示：")
    st.image("src/modelquantize/datatype.png",caption="fig.1:数据类型")
    st.markdown("&emsp;&emsp;具体而言，模型量化是一种压缩网络参数的方式，它将神经网络的权重($\\text{weight}$)、特征图($\\text{activation}$)等原本用浮点表示的量值换用定点（整型）表示，在计算过程中，再将定点数据反量化回浮点数据，得到结果。由于将数据类型量化再反量化，因此会有一定精度损失，但如果量化位数合适，基本不会对\
        网络性能造成较大影响。")
    with st.expander("数值类型拓展讲解:"):
        st.markdown("数值类型")
        pass
    st.markdown("#### :blue[对哪些数值进行量化？]")
    st.markdown("&emsp;&emsp;模型量化对象主要包括以下几个方面：")
    st.markdown("&emsp;&emsp;:blue[$\\text{Weight}$]：就是模型的参数。")
    st.markdown("&emsp;&emsp;:blue[$\\text{Activation}$]：激活值，模型的激活值往往存在异常值，直接对其做量化会导致严重的精度损失，因此一般需要比较复杂的处理方法。")
    st.markdown("&emsp;&emsp;:blue[$\\text{Gradients}$]：训练深度学习模型时，梯度通常是浮点数，它主要作用是在分布式计算中减少通信开销，同时，也可以减少反向传播时的开销。")
    
    st.markdown("#### :blue[量化方法如何分类？]")
    st.markdown("&emsp;&emsp;根据量化方案的不同，可以分为量化感知训练($\\text{QAT}$)和后训练量化($\\text{PTQ}$)。")
    st.markdown("&emsp;&emsp;:blue[$\\text{QAT(Quant-Aware Training)}$]：")
    st.markdown("&emsp;&emsp;:blue[$\\text{PTQ(Post Training Quantization)}$]：训练后量化也可以分成两种，权重量化和全量化。参考自[[1]](https://zhuanlan.zhihu.com/p/662881352)")
    st.markdown("&emsp;&emsp;&emsp;&emsp;$\\text{权重量化}$：权重量化仅量化模型的权重以压缩模型的大小，在推理时将权重反量化为原始的$\\text{float32}$数据，后续推理流程与普通的$\\text{float32}$模型一致。\
        权重量化的好处是不需要校准数据集，不需要实现量化算子，且模型的精度误差较小，由于实际推理使用的仍然是$\\text{float32}$算子，所以推理性能不会提高。如:blue[$\\text{QLoRA}$]。")
    st.markdown("&emsp;&emsp;&emsp;&emsp;$\\text{全量化}$：全量化不仅会量化模型的权重，还会量化模型的激活值，在模型推理时执行量化算子来加快模型的推理速度。为了量化激活值，需要用户提供一定数量的校准数据集用于统计每一层激活值的分布，并对量化后的算子做校准。\
        校准数据集可以来自训练数据集或者真实场景的输入数据，需要数量通常非常小。如:blue[$\\text{GPTQ}$]")