import streamlit as st

def page2():
    st.title("Second page")

pg = st.navigation([
    st.Page("page_1.py", title="Learning to Rank", icon="📚"),
    st.Page("page_2.py", title="Memory Design in AI-Agents", icon="🧠"),
])
pg.run()