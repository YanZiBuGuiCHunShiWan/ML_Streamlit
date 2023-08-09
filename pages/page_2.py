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
st.latex(r'''\begin{aligned}{\tilde w}&=\arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}\lnp(y_i|x_i;w) \\ 
&= \arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}\ln(exp\left\{{-(\frac{(y_i-(w^Tx_i+b))^2}{2\sigma^2})}\right\}) \\ 
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
st.latex(r'''\begin{aligned}\ln{P}&=ln\prod_{i=1}^{n}{p(y_{i}=1|x_{i};w)^{y_{i}}*p(y_{i}=0|x_{i};w)^{1-y_{i}}} \\ 
&=\Sigma_{i=1}^{n}{y_{i}\ln{p}(y_{i}=1|x_{i};w)+(1-y_{i})\ln{p}(y_{i}=0|x_{i};w)} \\
&=\Sigma_{i=1}^{n}y_{i}\ln\cfrac{e^{w^Tx_{i}}}{1+e^{w^Tx_{i}}}+(1-y_{i})\ln\cfrac{1}{1+e^{w^Tx_{i}}}  \\
&=\Sigma_{i=1}^{n}y_{i}(w^Tx_{i}-\ln(1+e^{w^Tx_{i}}))-(1-y_{i})\ln(1+e^{w^Tx_{i}})  \\ 
&=\Sigma_{i=1}^{n}y_{i}w^Tx_{i}-\ln(1+e^{w^Tx_{i}})\end{aligned} \tag {1.2}''')
st.markdown("&emsp;&emsp;寻求的最优参数$w$就相当于找一个$w$使得似然函数最大,即：")
st.latex(r''' \begin{aligned}\tilde{w}&=\arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}~y_{i}w^Tx_{i}-\ln(1+e^{w^Tx_{i}}) \\
 &=\arg\mathop{\min}\limits_{w}\Sigma_{i=1}^{n}~\ln(1+e^{w^Tx_{i}})-y_{i}w^Tx_{i}\end{aligned} \tag{1.3}''')
st.markdown("&emsp;&emsp;而损失函数就是$loss=\Sigma_{i=1}^{n}~\ln(1+e^{w^Tx_{i}})-y_{i}w^Tx_{i}$。损失函数关于$w$求导公式如下:")
st.latex(r'''\begin{aligned} \displaystyle \frac{\partial L(w)}{\partial w}&=\Sigma_{i=1}^{n}~\ln\frac{e^{w^Tx_{i}}}{1+e^{w^Tx_{i}}}x_{i}-y_{i}x_{i} \\
&=\Sigma_{i=1}^{n}~y^{*}_{i}x_{i}-y_{i}x_{i} \\
&=\Sigma_{i=1}^{n}~(y^{*}_{i}-y_{i})x_{i} \end{aligned} \tag{1.4}''')

st.markdown("&emsp;&emsp;其中$y_{i}^*=\ln\cfrac{e^{w^Tx_{i}}}{1+e^{w^Tx_{i}}}$就是样本被归类为$1$的概率。\
      以上便是损失函数关于参数向量的一阶导数。使用梯度下降更新参数的公式如下：")
st.latex(r'''w^{new}=w^{old}-learningrate*\frac{\partial L(w)}{\partial w} \tag{1.5}''')
st.markdown("&emsp;&emsp;更进一步，可以尝试将公式用矩阵乘法的形式重新表达:")
st.latex(r'''\begin{aligned} 
\displaystyle \frac{\partial L(w)}{\partial w}&=\Sigma_{i=1}^{n}~(y^{*}_{i}-y_{i})x_{i} \\ 
&=(x_{1},x_{2},...,x_{n})(y_{1}^{*}-y_{1},y_{2}^{*}-y_{2},...,y_{n}^{*}-y_{n})^T \\
&= X^T(Y^*-Y) \end{aligned} \tag{1.6}''')

st.markdown("&emsp;&emsp;再看损失函数关于参数向量的二阶导，即海森矩阵：")
st.latex(r'''\begin{aligned} \displaystyle\frac{\partial L^{2}(w)} {\partial w \partial w^T}&=\Sigma_{i=1}^{n}~\frac{e^{w^Tx_{i}}}{1+e^{w^Tx_{i}}}x_{i}x_{i}^T \in R^{m \times m} \\ 
&=\Sigma_{i=1}^{n}x_{i}p(y_{i}=1|x_{i};w)x_{i}^T \\
&=x_{1}p(y_{1}=1|x_{1};w)x_{1}^T+x_{2}p(y_{2}=1|x_{2};w)x_{2}^T+....+x_{n}p(y_{n}=1|x_{n};w)x_{n}^T  \\ 
&=(x_1,x_2,...,x_n) \begin{pmatrix} { p(y_{1}=1|x_{1};w)} & {0 } & { \cdots } & { 0 } \\ { 0 } & { p(y_{2}=1|x_{2};w) } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { } & { \vdots } \\ { 0} & { 0 } & { \cdots } & {p(y_{n}=1|x_{n};w)} \end{pmatrix}(x_1,x_2,...,x_n)^T \\ 
&=X^TDiag(P)X \end{aligned}''')
st.markdown("&emsp;&emsp;$Diag(P)$对角线上的元素一定大于$0$，很容易证明该矩阵 $X^TDiag(P)X \succeq 0$,所以损失函数是一个凸函数，没有局部最优。") 
 
st.markdown("##### 恭喜你完成本章阅读，以下是考察你知识掌握情况的选择题，请耐心作答噢！✏️")
with st.expander('No.1 :blue[我们根据极大似然估计推导出了交叉熵损失，那么逻辑回归的损失是否可以选择平方损失函数呢]'):
      st.markdown("##### 一起揭晓吧")