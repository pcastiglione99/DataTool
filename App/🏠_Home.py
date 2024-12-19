import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(page_title="DataTool", page_icon="📊", initial_sidebar_state="collapsed", menu_items=None)
st.title('Welcome to DataTool 📊')

if st.button("Let's start!"):
    switch_page("upload")

