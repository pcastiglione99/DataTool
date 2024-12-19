import streamlit as st
import pandas as pd
from functools import partial
from dataquality import measures
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(
    page_title='DataTool|Analysis', 
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed', 
    menu_items=None
)

if 'compute_block' not in st.session_state:
    st.session_state['compute_block'] = False


df = st.session_state['df']
y_class = st.session_state['y_class']

measures_to_compute = {
    "Coverage": "The coverage metric is the measure of the degree to which the dataset is representative of the real-world", 
    "Density":"", 
    "Diversity":"", 
    "Class Overlap":"The Class Overlap measure assesses the presence of data points that belong to different classes, but lie close together. It is a measure of significant importance to estimate the difficulty of a classification task and consequently helps in implementing appropriate strategies or to address such challenges. ", 
    "Label Purity":"Label purity quantifies the occurrence of label errors or inconsistencies in a dataset. It is an estimation of the degree to which the labels of a dataset reflect the intended class assignments and it is of considerable importance to the success of a classification task.", 
    "Class Balance":"The class balance metric assesses how evenly distributed are the values of an attribute or the labels of a dataset. Evaluating the distribution of attribute values helps uncover any discrepancies in representation, while examining label distribution is crucial for understanding class balance.", 
    "Group Fairness":"The group fairness measure quantifies the difference in favorable outcomes between a privileged group and an unprivileged group. This measure aims to assess whether there is disparity in how outcomes are distributed among different demographic groups.", 
    "Bias Distance":"The bias distance measure quantifies the disparity between the distribution observed in the real-world population and the distribution represented within the dataset. It can be useful in identifying inconsistencies in data representation, thus enabling the identification of potential discrimination or bias.", 
    "Duplicates":"The Duplicates measure quantifies the fraction of duplicate rows in a dataset", 
    "Skeweness":"The Skeweness measure quantifies how different it is the distribution from its ideal Gaussian fitted distribution"
}


st.title("🔍 Analyis")

to_measure = st.selectbox(label = "Select a characteristic to measure", options=list(measures_to_compute.keys()))
st.caption(measures_to_compute[to_measure])
st.dataframe(df)


st.session_state['compute_block'] = False

if to_measure == "Coverage":
    st.session_state['compute_block'] = True
    columns = st.columns(2)
    with columns[0]:
        attribute = st.selectbox("Select the attribute", options=df.columns)
    with columns[1]:
        ground_truth = st.text_input("Insert the expected values")
        if ground_truth: st.session_state['compute_block'] = False
    ground_truth = ground_truth.split(",")
    fun = partial(measures.coverage_attr, df,attribute, ground_truth)

elif to_measure == "Density":
    x = st.radio("Select where to compute the measure", options=['Dataset', 'Attribute'])
    if x == 'Attribute':
        attribute = st.selectbox("Select the attribute", options=df.columns)
        fun = partial(measures.density_attr, df, attribute)
    else:
        fun = partial(measures.density_df, df)

elif to_measure == "Diversity":
    columns = st.columns(2)
    with columns[0]:
        x = st.radio("Select where to compute the measure", options=['Dataset', 'Attribute'])
    with columns[1]:
        y = st.radio("Select the method", options=['Shannon', 'Gini'])
    if x == 'Attribute':
        attribute = st.selectbox("Select the attribute", options=df.columns)
        if y == 'Shannon':
            fun = partial(measures.diversity_attr_shannon, df, attribute)
        else:
            fun = partial(measures.diversity_attr_gini, df, attribute)
    else:
        if y == 'Shannon':
            fun = partial(measures.diversity_df_shannon, df)
        else:
            fun = partial(measures.diversity_df_gini, df)
elif to_measure == "Class Overlap":
    threshold = st.slider("Select the threshold", min_value=0.01, max_value=0.1, value=0.03)
    fun = partial(measures.class_overlap, df, y_class, threshold, False)

elif to_measure == "Label Purity":
    fun = partial(measures.label_purity, df, y_class)

elif to_measure == "Class Balance":
    columns = st.columns(2)
    with columns[0]:
        attribute = st.selectbox("Select the attribute", options=df.columns)
    with columns[1]:
        method = st.radio("Select the method",options=['Absolute', 'Average'])
    fun = partial(measures.class_balance, df, attribute, method)

elif to_measure == "Group Fairness":
    columns = st.columns(3)
    with columns[0]:
        sensible_attribute = st.selectbox("Select the sensible attribute", options=df.columns)
    with columns[1]:
        privileged_value = st.selectbox("Select the privileged value", options=df[sensible_attribute].unique())
    with columns[2]:
        favorable_y = st.selectbox("Select the favorable_y", options=df[y_class].unique())
    fun = partial(measures.group_fairness, df, sensible_attribute, privileged_value, y_class, favorable_y)

elif to_measure == "Bias Distance":
    st.session_state['compute_block'] = True
    method = st.radio("Select the method", options=['Chi Square Test', 'Bhattacharyya'])
    columns = st.columns(2)
    with columns[0]:
        attribute = st.selectbox("Select the attribute", options=df.columns)
        observed = df[attribute].value_counts(normalize=True)
    with columns[1]:
        try:
            expected = st.text_input(label="Insert the expected distribution as shown",placeholder="a:0, b:1, c:0")
            expected = dict((key.strip(), float(value.strip())) for key, value in (pair.split(':') for pair in expected.split(',')))
            if sum(expected.values()) != 1:
                raise Exception("Not a valid distribution!")
        except:
            st.warning("Insert a valid distribution")
        else:
            st.session_state['compute_block'] = False
            fun = partial(measures.chi_square_test, observed, expected) if method=='Chi Square Test' else partial(measures.bhattacharyya_coef, observed, expected)

elif to_measure == "Duplicates":
    fun = partial(measures.duplicates, df)

elif to_measure == "Skeweness":
    data = measures.encode_features(df)
    data = measures.scale_features(data)
    columns = st.columns(2)
    with columns[0]:
        x = st.radio("Select where to compute the measure", options=['Dataset', 'Attribute'])
    with columns[1]:
        method = st.radio("Select the method", options=["Normalized Squared Error", 'Intersection Over Union'])
    if x == 'Attribute':
        attribute = st.selectbox("Select the attribute", options=[col for col in df.columns if df[col].dtype != 'O'])
        fun = partial(measures.skewness_attr, data, method, attribute)
    else:
        fun = partial(measures.skewness_df, data, method)

if 'btn_compute' not in st.session_state: 
    st.session_state['btn_compute'] = False

btn_compute = st.empty()
if btn_compute.button("🚀 Analyze!", disabled=st.session_state['compute_block']):
    st.session_state['btn_compute'] = not st.session_state['btn_compute']
    btn_compute.empty()
    with st.spinner("Please wait..."):
        if to_measure == "Skeweness":
            if x == "Attribute":
                measure, fig = fun()
                st.write(f'## {to_measure} = {measure}')
                st.pyplot(fig)
            else: 
                measure = fun()
                st.write(f'## {to_measure} = {measure}')
        else:
            measure = fun()
            st.write(f'## {to_measure} = {measure}')

if st.button("📋 Summary"):
    switch_page("summary")
