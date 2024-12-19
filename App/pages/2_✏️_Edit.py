import time
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from functools import partial
from dataquality import edit

st.set_page_config(
    page_title="DataTool|Edit", 
    page_icon="📊",
    layout='wide',
    initial_sidebar_state="collapsed", 
    menu_items=None)

df = st.session_state['df']
if 'df_new' not in st.session_state:
    st.session_state['df_new'] = df
y_class = st.session_state['y_class']
edit_operations = [
    "Attribute Density", "Dataset Density", "Class Overlap", 
    "Label Purity", "Class Balance", "Group Fairness",
    "Duplicates"
]

st.write("# ✏️ Edit")

to_edit = st.selectbox(label = "Select a characteristic to edit", options=edit_operations)

st.dataframe(df)

if to_edit == "Attribute Density":
    columns = st.columns(2)
    with columns[0]:
        attribute = st.selectbox("Select the attribute to edit", df.columns)
    with columns[1]:
        percentage = st.slider("Select the percentage of the edit", min_value=0.0, max_value=1.0, value=0.2)
    fun = partial(edit.edit_density_attr, df, attribute, percentage)

elif to_edit == "Dataset Density":
    percentage = st.slider("Select the percentage of the edit", min_value=0.0, max_value=1.0, value=0.2)
    fun = partial(edit.edit_density_df, df, y_class, percentage)

elif to_edit == "Class Overlap":
    columns = st.columns(3)
    with columns[0]:
        strategy = st.selectbox("Select the strategy", ['SMOTE','Undersampler','Random', 'ADASYN'])
    with columns[1]:
        if strategy != 'Undersampler':
            factor = st.slider("Select the factor of the edit", min_value=1, max_value=30, value=4)
        else:
            factor = st.slider("Select the factor of the edit", min_value=0.0, max_value=1.0, value=0.3)
    with columns[2]:
        threeshold_value = st.slider("Select the threeshold value", min_value=0.01, max_value=0.1, value=0.03)
    fun = partial(edit.edit_class_overlap, df, y_class, factor, threeshold_value, strategy)

elif to_edit == "Label Purity":
        factor = st.slider("Select the factor of the edit", min_value=0.0, max_value=1.0, value=0.5)
        fun = partial(edit.edit_label_purity, df, y_class, factor)

elif to_edit == "Class Balance":
        balance = st.slider("Select the desidered balance", min_value=0.0, max_value=1.0, value=0.5)
        fun = partial(edit.edit_class_balance, df, y_class, balance)

elif to_edit == "Group Fairness":
    columns = st.columns(4)
    with columns[0]:
        sensible_attribute = st.selectbox(label="Select the sensible attribute",options=df.columns)
    with columns[1]:
        privileged_value = st.selectbox(label="Select the privileged value", options=df[sensible_attribute].unique())
    with columns[2]:
        favorable_outcome = st.selectbox(label="Select the favorable outcome", options=df[y_class].unique())
    with columns[3]:
        balance = st.slider("Select the fraction of rows to remove", min_value=0.0, max_value=1.0, value=0.5)
    fun = partial(edit.edit_group_fairness, df, y_class, favorable_outcome ,sensible_attribute, privileged_value, balance)
elif to_edit == "Duplicates":
    percentage = st.slider("Select the desidered percentage of duplicated rows", min_value=0.0, max_value=1.0, value=0.5)
    fun = partial(edit.edit_duplicates, df, percentage)

#if 'btn_compute' not in st.session_state: st.session_state['btn_compute'] = False
btn_compute = st.empty()
if btn_compute.button("Edit!"):
    #st.session_state['btn_compute'] = not st.session_state['btn_compute']
    btn_compute.empty()
    with st.spinner("Please wait..."):
        df_new=fun()
        st.session_state['df_new'] = df_new
    st.dataframe(df_new)
    st.download_button(
        label = "Download edited data", 
        data = df_new.to_csv(index=False),
        file_name = f"{st.session_state['filename']}_new.csv",
        mime='text/csv'
    )

#if st.session_state['btn_compute']:
    #st.session_state['btn_compute'] = not st.session_state['btn_compute']
if st.button("🔍 Start the analysis"):
    st.session_state['df'] = st.session_state['df_new']
    switch_page('analysis')



