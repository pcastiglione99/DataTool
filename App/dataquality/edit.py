import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from .measures import class_overlap

def drop_col(df, to_remove):
    df_new = df.drop(to_remove, axis=1)
    return df_new

def edit_density_attr(df, attribute, percentage):
    vc = df[attribute].value_counts()
    indexes = df[df[attribute] == vc.index[1]].sample(frac=percentage, random_state=42).index
    df_new = df.drop(indexes).reset_index(drop=True)
    return df_new

def edit_density_df(df, y_class, percentage):
    df_new = df.copy()
    for f in df.columns:
        if f == y_class: continue
        if df[f].nunique() <= 6 and df[f].nunique() >= 2:
            df_new = edit_density_attr(df_new, f, percentage)
    return df_new.reset_index(drop=True)


def edit_class_overlap(df, y_class, factor, threshold_value=0.03, strategy='SMOTE'):
    X = df.drop(y_class, axis=1)
    y = df[y_class] 
    points_inside_boundary_X = class_overlap(df, y_class, threshold_value=threshold_value, return_points=True)
    points_inside_boundary_y = y.iloc[points_inside_boundary_X.index]

    points_outside_boundary_X = X.loc[X.index.difference(points_inside_boundary_X.index)]
    points_outside_boundary_y = y.loc[X.index.difference(points_inside_boundary_X.index)]

    if strategy == 'Undersampler':
        undersampler = RandomUnderSampler(
            sampling_strategy={value: int(count * factor) for value, count in points_outside_boundary_y.value_counts().items()},
            random_state=42
        )
        X_under, y_under = undersampler.fit_resample(points_outside_boundary_X, points_outside_boundary_y)
        X_new = pd.concat([X_under, points_inside_boundary_X]).reset_index(drop=True)
        y_new = pd.concat([y_under, points_inside_boundary_y]).reset_index(drop=True)
    else:
        if strategy == 'SMOTE':
            oversampler = SMOTE(
                sampling_strategy={value: int(count * factor) for value, count in points_inside_boundary_y.value_counts().items()},
                random_state=42
            )
        elif strategy == 'Random':
            oversampler = RandomOverSampler(
                sampling_strategy=dict(points_inside_boundary_y.value_counts() * factor),
                random_state=42
            )
        elif strategy == 'ADASYN':
            oversampler = ADASYN(
                sampling_strategy={value: int(count * factor) for value, count in points_inside_boundary_y.value_counts().items()},
                random_state=42
            )
        else:
            raise ValueError("Invalid strategy. Supported strategies are: 'SMOTE', 'Random', 'ADASYN', 'Undersampler'")

        X_over, y_over = oversampler.fit_resample(points_inside_boundary_X, points_inside_boundary_y)
        X_new = pd.concat([X.drop(points_inside_boundary_X.index), X_over]).reset_index(drop=True)
        y_new = pd.concat([y.drop(points_inside_boundary_X.index), y_over]).reset_index(drop=True)
    df_new = X_new
    df_new[y_class] = y_new

    return df_new

def edit_label_purity(df, y_class, frac):
    
    df_new = df.copy()
    
    outcomes = list(df[y_class].unique())
    print(outcomes)
    for idx in df.sample(frac=frac, random_state=42).index:
        df_new.at[idx, y_class] = outcomes[1 - outcomes.index(df.at[idx, y_class])]
    
    return df_new

def edit_class_balance(df, y_class, balance):
    data0 = df.loc[df[y_class] == 0]
    data1 = df.loc[df[y_class] == 1]

    current_ratio = len(data1) / len(data0)

    combined_data = None

    if balance < current_ratio:
        combined_data = pd.concat([data0, data1[:round(len(data0) * balance)]])
    else:
        combined_data = pd.concat([data0[:round(len(data1) / balance)], data1])

    combined_data = combined_data.sample(frac=1, random_state=42)

    return combined_data

def edit_group_fairness(df, y_class, favorable_outcome, sensible_attribute, privileged_value, balance):
    df_new = df.drop(df[(df[sensible_attribute] == privileged_value) & (df[y_class] == favorable_outcome)].sample(frac=balance, random_state=42).index)
    return df_new

def edit_duplicates(df, percentage):
    df_new = pd.concat([df, df.sample(frac=percentage, random_state=42)]).sample(frac=1,  random_state=42).reset_index(drop=True)
    return df_new
