import os
import streamlit as st
#左边侧栏的显示
st.sidebar.markdown("## 逻辑回归 ❄️")
st.write("Welcome to my 3 space~")

st.markdown("## 1.0线性回归")
st.latex(r'''X=(x_1,x_2,...,x_n)^T=
         \begin{bmatrix} 
   a & b \\ 
   c & d 
\end{bmatrix}''')

st.latex("\sum_{i=1}^{n}")
st.latex(r'''\begin{aligned}{\tilde w}&=\arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}lnp(y_i|x_i;w) \\ 
&= \arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}ln(exp\left\{{-(\frac{(y_i-(w^Tx_i+b))^2}{2\sigma^2})}\right\}) \\ 
&=\arg \mathop{\min}\limits_{w}\Sigma_{i=1}^{n}\frac{(y_i-(w^Tx_i+b))^2}{2\sigma^2} \\ 
&\propto \arg\mathop{\min}\limits_{w}\Sigma_{i=1}^{n}(y_i-(w^Tx_i+b))^2\end{aligned} ''')

####################################逻辑回归##########################################
st.markdown("## 1.1逻辑回归")
####################################逻辑回归模型######################################
st.markdown("### 1.1.1 逻辑回归模型")
st.markdown("&emsp;&emsp;逻辑回归是用来做分类算法的。把线性回归的结果$Y$代入一个非线性变换的$ sigmoid$函数中，\
      即可得到$[0,1]$之间取值范围的数$S$，$S$可以把它看成是一个概率值，如果设置概率阈值为$0.5$，那么$S$大于$0.5$可以看成是正样本，\
            小于$0.5$看成是负样本，就可以进行分类。$sigmoid$函数公式如下：")
st.latex(r'''\begin{aligned}sigmoid(x)=\frac{1}{1+e^{-x}}\end{aligned} \tag {1.1}''')
st.markdown("&emsp;&emsp;在二分类逻辑回归场景下，\
            给定模型参数$w_i$和样本参数$x_i$,\
            那么样本被分类为1的概率为$p(y_i=1|x_i;w)=\cfrac{1}{1+e^{-w^T{x_i}}}=\cfrac{e^{w^Tx_i}}{1+e^{w^T{x_i}}}$。而单个样本的标签分类最终只能是$0$或者$1$。")
st.markdown("&emsp;&emsp;因此，可以用公式$p(y_{i}=1|x_{i};w)^{y_{i}}*p(y_{i}=0|x_{i};w)^{1-y_{i}}$对样本的预测信息进行统一描述，当$y_{i}=1$时，\
            公式退化为$p(y_{i}=1|x_{i};w)$,当$y_{i}=0$时，公式退化为$p(y_{i}=0|x_{i};w)$。根据极大似然估计，将样本的对数似然最大化，先写出对数似然函数:")
st.latex(r'''\begin{aligned} lnP&=ln\prod_{i=1}^{n}{p(y_{i}=1|x_{i};w)^{y_{i}}*p(y_{i}=0|x_{i};w)^{1-y_{i}}} \\ 
&=\Sigma_{i=1}^{n}{y_{i}lnp(y_{i}=1|x_{i};w)+(1-y_{i})lnp(y_{i}=0|x_{i};w)} \\
&=\Sigma_{i=1}^{n}y_{i}ln\cfrac{e^{w^Tx_{i}}}{1+e^{w^Tx_{i}}}+(1-y_{i})ln\cfrac{1}{1+e^{w^Tx_{i}}}  \\
&=\Sigma_{i=1}^{n}y_{i}(w^Tx_{i}-ln(1+e^{w^Tx_{i}}))-(1-y_{i})ln(1+e^{w^Tx_{i}})  \\ 
&=\Sigma_{i=1}^{n}y_{i}w^Tx_{i}-ln(1+e^{w^Tx_{i}})\end{aligned} \tag {1.2}''')
st.markdown("&emsp;&emsp;寻求的最优参数$w$就相当于找一个$w$使得似然函数最大,即：")
st.latex(r''' \begin{aligned}\tilde{w}&=\arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}~y_{i}w^Tx_{i}-ln(1+e^{w^Tx_{i}}) \\
 &=\arg\mathop{\min}\limits_{w}\Sigma_{i=1}^{n}~ln(1+e^{w^Tx_{i}})-y_{i}w^Tx_{i}\end{aligned} \tag{1.3}''')
