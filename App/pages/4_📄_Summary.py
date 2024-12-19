import streamlit as st
import numpy as np
import pandas as pd
from dataquality import measures
import altair as alt

st.set_page_config(
    page_title='DataTool|Summary', 
    page_icon='📋',
    layout='wide',
    initial_sidebar_state='collapsed', 
    menu_items=None
)
st.markdown('''
            <style>
            div[data-testid="stHorizontalBlock"] {
                box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
                border-radius: 20px;
                padding:20px;
                padding-top:60px;
            }
            </style>
            ''',
            unsafe_allow_html=True)
df = st.session_state['df']
y_class = st.session_state['y_class']

st.title("Summary")
with st.container():
    st.write("## Density")
    cols = st.columns(2)
    with cols[0]:
        density = round(measures.density_df(df), 3)
        st.markdown(f'#### <div style="text-align: center; font-size:100px; margin-top:20px">{density}', unsafe_allow_html=True)
        if density <= 0.6:
            st.warning("Density is low")
        elif density < 0.8:
            st.warning("Density is acceptable")
        else:
            st.success("Density is good!")
    with cols[1]:
        density_attrs = pd.DataFrame({
            "Density": [measures.density_attr(df, attr) for attr in df.columns],
            "Feature": list(df.columns)
        })
        color_scale = alt.Scale(range=['#FFB0B0','#FF7D7D','#FF4B4B'])
        bars = alt.Chart(density_attrs).mark_bar().encode(
            x='Feature',
            y='Density',
            color=alt.Color('Density:Q', scale=color_scale, legend=None)
        )
        st.altair_chart(bars, use_container_width=True)


with st.container():
    #st.markdown('## <div style="text-align: right;">Diversity</div>', unsafe_allow_html=True)
    st.markdown('## Diversity')
    cols = st.columns(2)
    with cols[1]:
        diversity = round(measures.diversity_df_gini(df), 2)
        st.markdown(f'#### <div style="text-align: center; font-size:100px; margin-top:20px">{diversity}', unsafe_allow_html=True)
        if diversity <= 0.6:
            st.warning("Diversity is low")
        elif density < 0.8:
            st.warning("Diversity is acceptable")
        else:
            st.success("Diversity is good!")
    with cols[0]:
        #st.write(f"#### Features Diversity:")
        diversity_attrs = pd.DataFrame({
            "Diversity": [measures.diversity_attr_gini(df, attr) for attr in df.columns],
            "Feature": list(df.columns)
        })
        color_scale = alt.Scale(range=['#FFB0B0','#FF7D7D','#FF4B4B'])
        bars = alt.Chart(diversity_attrs).mark_bar().encode(
            x='Feature',
            y='Diversity',
            color=alt.Color('Diversity:Q', scale=color_scale, legend=None)
        )
        st.altair_chart(bars, use_container_width=True)

st.write("## Class Balance")
cols = st.columns(2)
with cols[0]:
    #st.markdown(f'## <div style="text-align: center; font-size:100px; margin-top:20px">{round(np.mean([measures.class_balance(df, attr, "Average") for attr in df.columns]),2)}', unsafe_allow_html=True)
    class_balance = round(measures.class_balance(df, y_class, "Average"),2)
    st.markdown(f'## <div style="text-align: center; font-size:100px; margin-top:20px">{class_balance}', unsafe_allow_html=True)
    if class_balance <= 0.3:
        st.error("Dataset is very imbalance")
    elif class_balance < 0.7:
        st.warning("Dataset is imbalance")
    else:
        st.success("Dataset is balanced")
with cols[1]:
    #st.write(f"#### Features Balance:")
    balance_attrs = pd.DataFrame({
        "Balance": [measures.class_balance(df, attr, 'Average') for attr in df.columns],
        "Feature": df.columns
    })
    color_scale = alt.Scale(range=['#FFB0B0','#FF7D7D','#FF4B4B'])
    bars = alt.Chart(balance_attrs).mark_bar().encode(
        x='Feature',
        y='Balance',
        color=alt.Color('Balance:Q', scale=color_scale, legend=None)
    )
    st.altair_chart(bars, use_container_width=True)

st.write("## Class Overlap")
columns = st.columns(1)
class_overlap = round(measures.class_overlap(df,st.session_state['y_class']), 2)
with columns[0]:
    #st.write(f"#### Class Overlap: {round(measures.class_overlap(df,st.session_state['y_class']), 3)}")
    st.markdown(f'## <div style="text-align: center; font-size:100px;">{class_overlap}', unsafe_allow_html=True)
    if class_overlap <= 0.3:
        st.success("Dataset has a low class overlap")
    elif class_overlap < 0.6:
        st.warning("Dataset has a acceptable class overlap")
    else:
        st.error("Dataset has an high class overlap!")

st.write("## Label Purity")
columns = st.columns(1)
with columns[0]:
    label_purity = round(measures.label_purity(df,st.session_state['y_class']),2)
    st.markdown(f'## <div style="text-align: center; font-size:100px;">{label_purity}', unsafe_allow_html=True)
    if label_purity <= 0.3:
        st.warning("Labels purity is bad!")
    elif label_purity < 0.6:
        st.warning("Dataset has a acceptable label purity")
    else:
        st.success("Dataset has a good label purity")


st.write("## Duplicates")
columns = st.columns(1)
with columns[0]:
    duplicates = round(measures.duplicates(df), 2)
    st.markdown(f'## <div style="text-align: center; font-size:100px;">{duplicates}', unsafe_allow_html=True)
    if duplicates <= 0.05:
        st.success("Dataset has a very low percentage of duplictes")
    elif duplicates < 0.1:
        st.warning("Dataset has an acceptable percentage of duplictes")
    else:
        st.error("Dataset has an high percentage of duplictes")




st.write("## Group Fairness")
with st.container():
    columns = st.columns(2)
    with columns[0]:
        sensible_attribute = st.selectbox("Select the sensible attribute", options=df.columns)
        privileged_value = st.selectbox("Select the privileged value", options=df[sensible_attribute].unique())
        favorable_y = st.selectbox("Select the favorable outcome", options=df[y_class].unique())
        if st.button("Compute","btn_fairness"):
            fairness = measures.group_fairness(df, sensible_attribute, privileged_value, y_class, favorable_y)
    with columns[1]:
        if 'fairness' in vars():
            st.markdown(f'## <div style="text-align: center; font-size:100px; margin-top:20px">{round(fairness,2)}', unsafe_allow_html=True)
            if fairness < 0.8:
                st.warning(f"The Dataset shows bias toward the value {privileged_value}")
            elif fairness > 1.2:
                st.warning(f"The Dataset discriminate the value {privileged_value}")
            else:
                st.success("The Dataset is fair")


st.write("## Coverage")
columns = st.columns(2)
with columns[0]:
    attribute = st.selectbox("Select the attribute", options=df.columns)
    ground_truth = st.text_input("Insert the expected values")
    ground_truth = ground_truth.split(",")
    if st.button("Compute","btn_coverage"):
        coverage = measures.coverage_attr(df, attribute, ground_truth)
    with columns[1]:
        if 'coverage' in vars():
            st.markdown(f'## <div style="text-align: center; font-size:100px; margin-top:20px">{round(coverage,2)}', unsafe_allow_html=True)

st.write("## Bias")
columns = st.columns(2)
with columns[0]:
    attribute = st.selectbox("Select the attribute",key='bias', options=df.columns)
    observed = df[attribute].value_counts(normalize=True)
    try:
        expected = st.text_input(label="Insert the expected distribution as shown",placeholder="a:0, b:1, c:0")
        expected = dict((key.strip(), float(value.strip())) for key, value in (pair.split(':') for pair in expected.split(',')))
        if sum(expected.values()) != 1:
            raise Exception("Not a valid distribution!")
    except:
        st.warning("Insert a valid distribution")

    if st.button("Compute","btn_bias"):
        bias_dist = measures.bhattacharyya_coef(observed, expected)
    with columns[1]:
        if 'bias_dist' in vars():
            st.markdown(f'## <div style="text-align: center; font-size:100px; margin-top:20px">{round(bias_dist, 2)}', unsafe_allow_html=True)
