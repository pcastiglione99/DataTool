import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import chi2_contingency, wasserstein_distance, norm, entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simps
import streamlit as st


@st.cache_data
def scale_features(X):
    X_new = X.copy()
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X_new), columns=X.columns)

@st.cache_data
def encode_features(X):
    X_new = X.copy()
    enc = LabelEncoder()
    for i in X_new.columns:
        if X_new[i].dtype == "object":
            X_new[i]=enc.fit_transform(X_new[i])
    return X_new



@st.cache_data
def coverage_attr(df, attribute, ground_truth):
    coverage = df[attribute].nunique() / len(ground_truth)
    return coverage

@st.cache_data
def density_attr(df, attribute):
    density_values = np.array(df[attribute].value_counts(normalize=True))
    value_avg = density_values.mean()
    return 1 - np.sqrt((density_values - value_avg) ** 2).mean()

@st.cache_data
def density_df(df):
    return np.mean([density_attr(df, attr) for attr in df.columns])


@st.cache_data
def diversity_attr_shannon(df, attribute):
    p = np.array(df[attribute].value_counts(normalize=True))
    return - (p * np.log2(p)).sum()

@st.cache_data
def diversity_df_shannon(df):
    return np.mean([diversity_attr_shannon(df, attr) for attr in df.columns])


@st.cache_data
def diversity_attr_gini(df, attribute):
    p = np.array(df[attribute].value_counts(normalize=True))
    return 1 - (p ** 2).sum()

@st.cache_data
def diversity_df_gini(df):
    return np.mean([diversity_attr_gini(df, attr) for attr in df.columns])

@st.cache_data
def class_overlap(df, y_class, threshold_value=0.05, return_points=False):
    df = df.dropna()
    X = df.drop(y_class, axis=1)
    y = df[y_class]

    X = encode_features(X)

    X = scale_features(X)
    

    model = LogisticRegression(penalty=None, max_iter=5000, random_state=42)
    model.fit(X, y)
    decision_values = model.decision_function(X)
    threshold = decision_values.ptp() * threshold_value
    inside_boundary_indices = np.where((decision_values >= -threshold) & (decision_values <= threshold))[0]
    if return_points:
        points_inside_boundary = X.iloc[inside_boundary_indices]
        return points_inside_boundary
    else:
        n_points_inside_boundary = len(inside_boundary_indices)
        overlap_percentage = n_points_inside_boundary / len(X)
        return overlap_percentage

@st.cache_data
def label_purity(df, y_class):
    df = df.dropna()
    X = df.drop(y_class, axis=1)
    y = df[y_class]

    X = encode_features(X)
    X = scale_features(X)
    
    n_clusters = y.nunique()
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    y_pred = kmeans.fit_predict(X)
    contingency_mat = contingency_matrix(y, y_pred)
    label_purity_score = ((np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)) - 0.5) * 2
    return label_purity_score


@st.cache_data
def class_balance(df, attribute, method='Average'):
    category_counts = df[attribute].value_counts()
    majority_class_count = category_counts.iloc[0]
    minority_counts = category_counts.iloc[1:]
    if method == 'Absolute': minority_counts=minority_counts.values[0]
    minority_count_average = np.mean(minority_counts)
    return minority_count_average / majority_class_count


@st.cache_data
def group_fairness(df, sensible_attribute, privileged_value, y_class, favorable_y):
    df_new = df.copy()
    df_new[y_class] = np.where(df_new[y_class]==favorable_y, 1, 0)
    avg_non_privileged = df_new[df_new[sensible_attribute] != privileged_value][y_class].mean()
    avg_privileged = df_new[df_new[sensible_attribute] == privileged_value][y_class].mean()
    fairness_metric = avg_non_privileged / avg_privileged
    return fairness_metric


@st.cache_data
def chi_square_test(observed, expected):
    if len(observed) > len(expected):
        expected = {k: expected[k] if k in expected and expected[k] is not None else 0 for k in observed}
    else:
        observed = {k: observed[k] if k in observed and observed[k] is not None else 0 for k in expected}
    eps = 1e-5
    observed_frequencies = np.array([(100 * value) + eps for key, value in observed.items()])
    expected_frequencies = np.array([(100 * value) + eps for key, value in expected.items()])
    chi2, p, _, _ = chi2_contingency([observed_frequencies, expected_frequencies])
    if p < 0.05:
       return "The difference is statistically significant!"
    else:
        return "The difference is not statistically significant."


@st.cache_data
def bhattacharyya_coef(observed, expected):
    if len(observed) > len(expected):
        expected = {k: expected[k] if k in expected and expected[k] is not None else 0 for k in observed}
    else:
        observed = {k: observed[k] if k in observed and observed[k] is not None else 0 for k in expected}
    eps = 1e-5
    observed_frequencies = np.array([(value) + eps for key, value in observed.items()])
    expected_frequencies = np.array([(value) + eps for key, value in expected.items()])
    bhattacharyya_coefficient = np.sum([np.sqrt(observed_frequencies[i] * expected_frequencies[i]) for i in range(len(expected_frequencies))])
    #return -np.log(bhattacharyya_coefficient)
    return round(1 - bhattacharyya_coefficient,4) + 0 

@st.cache_data
def duplicates(df):
    return df.duplicated().sum() / df.shape[0]

'''
@st.cache_data
def skewness_attr(df, method, attr, show_fig=True):
    data = df[attr].dropna()
    #plt.hist(data, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black')
    kde = sns.kdeplot(data, gridsize = 100, label='KDE')
    mu, std = norm.fit(data)
    gaussian = norm(mu, std)
    x = np.linspace(min(data), max(data), 100)
    gaussian_data=np.random.normal(mu, std, len(data))
    kde_values = kde.get_lines()[0].get_ydata()
    gaussian_values = gaussian.pdf(x)

    kde_values /= simps(kde_values, dx=x[1] - x[0])
    gaussian_values /= simps(gaussian_values, dx=x[1] - x[0])
    if method == 'Kullback-Leibler':
        dist = entropy(kde_values, gaussian_values)
    else: 
        dist = wasserstein_distance(kde_values, gaussian_data)

    if show_fig:
        fig, ax = plt.subplots()
        sns.kdeplot(data, gridsize = 100, label='KDE', ax=ax)
        ax.hist(gaussian_data, bins=50, density=True, alpha=0.5, color='red', edgecolor='black')
        ax.plot(x, gaussian.pdf(x), label='Gaussian')
        ax.hist(data, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black')
        return dist, fig
    else:
        return dist
'''

def histogram_intersection(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

def histogram_union(hist1, hist2):
    return np.maximum(hist1, hist2).sum()

def histogram_intersection_over_union(hist1, hist2):
    intersection = histogram_intersection(hist1, hist2)
    union = histogram_union(hist1, hist2)
    iou = intersection / union
    return iou

@st.cache_data
def skewness_attr(df, method, attr, show_fig=True):
    data = df[attr].dropna()
    b = 50
    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(data)
    gaussian = norm(mu, std)
    gaussian_data=np.random.normal(mu, std, len(data))
    
    bin_edges = np.linspace(min(data), max(data), num=b)
    h1, _ = np.histogram(data, bins=bin_edges, density=True)
    h2, _ = np.histogram(gaussian_data, bins=bin_edges, density=True)

    mse = (np.square(h1 - h2)).mean(0)
    nmse = mse/np.var(h1)

    iou = 1 - histogram_intersection_over_union(h1, h2)
    
    dist = nmse if method == 'Normalized Mean Squared Error' else iou

    # Plot the histogram of the data and the gaussian fit
    if show_fig:
        fig, ax = plt.subplots()
        sns.kdeplot(data, gridsize = b, label='KDE', ax=ax)
        ax.hist(gaussian_data, bins=bin_edges, density=True, alpha=0.5, color='red', edgecolor='black')
        ax.plot(bin_edges, gaussian.pdf(bin_edges), label='Gaussian')
        ax.hist(data, bins=bin_edges, density=True, alpha=0.5, color='blue', edgecolor='black')
        return dist, fig
    else:
        return dist


@st.cache_data
def skewness_df(df, method):
    distances = []
    for attr in [col for col in df.columns if df[col].dtype != 'O']:
        dist = skewness_attr(df,method, attr, False)
        distances.append(dist)
    return np.mean(distances)

