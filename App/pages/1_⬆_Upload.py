import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd

st.set_page_config(page_title="DataTool|Upload", 
                   page_icon="📊", 
                   initial_sidebar_state="collapsed", 
                   menu_items=None)

if 'df' not in st.session_state:
    st.session_state['df'] = None

@st.cache_data
def read_csv(uploaded_file):
    if uploaded_file is not None:
        st.session_state['filename'] = str(uploaded_file.name[:uploaded_file.name.rfind('.')])
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.success("File uploaded correctly!", icon='✅')
            return df
        except:
            st.error("An error has occurred. Please try again", icon='🚨')

st.write("## 📁Upload your dataset:")

def new_file_selected():
    st.session_state['df'] = None

uploaded_file = st.file_uploader("Choose a CSV file", on_change=new_file_selected)

if st.session_state['df'] is None:
    df = read_csv(uploaded_file)
    st.session_state['df'] = df

if st.session_state['df'] is not None:
    df = st.session_state['df']
    st.dataframe(df)
    y_class = st.selectbox("Select the outcome column", df.columns, len(df.columns) - 1)
    st.session_state['y_class'] = y_class
    to_remove = st.multiselect("Select the columns to remove", df.columns)
    remove_btn = st.empty()
    if to_remove != []: 
        if remove_btn.button("Remove!", disabled=(not to_remove)):
            df = df.drop(to_remove, axis=1)
            st.session_state['df'] = df
            st.dataframe(df)
    columns = st.columns([0.35,0.35,0.22])
    with columns[0]:
        btn_analysis = st.button('🔍 Start the analysis')
        if btn_analysis:
            st.session_state['df'] = df
            switch_page('analysis')
    with columns[1]:
        btn_analysis = st.button('📋 Go to Summary')
        if btn_analysis:
            st.session_state['df'] = df
            switch_page('summary')
    with columns[2]:
        btn_edit = st.button('✏️ Edit the dataset')
        if btn_edit:
            st.session_state['df'] = df
            switch_page('edit')

