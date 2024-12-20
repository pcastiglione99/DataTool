import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(page_title="DataTool", page_icon="📊", initial_sidebar_state="collapsed", menu_items=None)
with st.sidebar:
    "[![Open in GitHub]](https://github.com/pcastiglione99/DataTool)"

st.title('Welcome to DataTool 📊')
st.caption("DataTool is an intuitive and user-friendly application designed for exploring, analyzing, and enhancing datasets through newly identified metrics")



if st.button("Let's start!"):
    switch_page("upload")

