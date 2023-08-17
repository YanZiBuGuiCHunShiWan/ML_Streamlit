import os
import streamlit as st
#左边侧栏的显示
#########################################符号约定################################################

st.markdown("# 线性回归模型")
st.sidebar.markdown("# 线性回归模型")
with st.expander("在正式学习算法课程之前，我们要先明确一下符号的定义，以免在后继学习中混淆概念~"):
      st.markdown("##### 样本矩阵与向量的符号定义如下：")
      st.markdown("###### &emsp;&emsp;样本矩阵:```有n个样本，每个样本有m维特征```")
      st.latex(r'''X=\begin{pmatrix}\mathbf {x_1} & \mathbf {x_2}&{\cdots}&\mathbf {x_n}\end{pmatrix}^T=\begin{pmatrix} \mathbf x_1^T \\  \mathbf x_2^T \\ {\vdots} \\ \mathbf x_n^T \end{pmatrix} \
            = \begin{pmatrix} x_{11} & x_{12} & {\cdots} & x_{1m} \\  x_{21} & x_{22} & {\cdots} & x_{2m}\\ {\vdots} &{\vdots} &{}&{\vdots} \\x_{n1} & x_{n2} & {\cdots} & x_{nm} \end{pmatrix}\
                  \in R^{n\times m}''')
      st.markdown("###### &emsp;&emsp;样本向量")
      st.latex(r'''\mathbf {x_i}=\begin{pmatrix} x_{11} & x_{12}&{\cdots}&x_{1m}\end{pmatrix}^T=\begin{pmatrix} x_{i1} \\  x_{i2} \\ {\vdots} \\ x_{in} \end{pmatrix}\in R^{m\times 1} ''')
      st.markdown("###### &emsp;&emsp;参数向量")
      st.latex(r'''\mathbf w=\begin{pmatrix} w_{1} & w_{2}&{\cdots}&w_{m}\end{pmatrix}^T=\begin{pmatrix} w_{1} \\  w_{2} \\ {\vdots} \\ w_{m} \end{pmatrix}\in R^{m\times 1} ''')
      st.markdown("###### &emsp;&emsp;标签向量")
      st.latex(r'''\mathbf y=\begin{pmatrix} y_{1} & y_{2}&{\cdots}&y_{n}\end{pmatrix}^T=\begin{pmatrix} y_{1} \\  y_{2} \\ {\vdots} \\ y_{n} \end{pmatrix}\in R^{n\times 1} ''')
      

st.markdown("## 1.0线性回归")
st.latex(r'''X=(x_1,x_2,...,x_n)^T=
         \begin{bmatrix} 
   a & b \\ 
   c & d 
\end{bmatrix}''')

st.latex("\sum_{i=1}^{n}")
st.latex(r'''\begin{aligned}{\tilde w}&=\arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}\ln{p}(y_i|x_i;w) \\ 
&= \arg \mathop{\max}\limits_{w}\Sigma_{i=1}^{n}\ln(exp\left\{{-(\frac{(y_i-(w^Tx_i+b))^2}{2\sigma^2})}\right\}) \\ 
&=\arg \mathop{\min}\limits_{w}\Sigma_{i=1}^{n}\frac{(y_i-(w^Tx_i+b))^2}{2\sigma^2} \\ 
&\propto \arg\mathop{\min}\limits_{w}\Sigma_{i=1}^{n}(y_i-(w^Tx_i+b))^2\end{aligned} ''')

