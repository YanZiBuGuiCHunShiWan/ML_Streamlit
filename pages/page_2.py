import os
import streamlit as st



st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# # Page 2 ❄️")
st.write("Welcome to my 3 space~")
st.markdown("# 机器学习")
st.markdown("## 线性回归")
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
