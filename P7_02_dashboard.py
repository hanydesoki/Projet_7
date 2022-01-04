import os

from data_science.data_management import DataGroup
from custom_preprocessing import filter_data


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# %%

# Loading data

@st.cache
def load_data(folder, n_sample=None):

    SAMPLE_DATA = n_sample is not None

    all_datas = DataGroup(folder)

    application_df = all_datas['application_train']
    bureau = all_datas['bureau']
    bb = all_datas['bureau_balance']
    prev = all_datas['previous_application']
    pos = all_datas['POS_CASH_balance']
    ins = all_datas['installments_payments']
    cc = all_datas['credit_card_balance']

    application_df = application_df[application_df['CODE_GENDER'] != 'XNA']

    if SAMPLE_DATA:
        application_df = application_df.sample(n_sample, random_state=0)

        # Filter data to match application
        bureau, bb, prev, pos, ins, cc = filter_data(application_df, bureau, bb, prev, pos, ins, cc)


    return application_df, bureau, bb, prev, pos, ins, cc


# %%

application_df, bureau, bb, prev, pos, ins, cc = load_data('Datas', n_sample=10000)


# %%

def filter_df(df, col, classes):
    return df[df[col].isin(classes)]

# %%

st.sidebar.header('Options')

gender_option = st.sidebar.selectbox('Genre', ['Tout'] + list(application_df['CODE_GENDER'].unique()))
target_option = st.sidebar.selectbox('Crédit', ['Tout', 'Accepté', 'Refusé'])
car_option = st.sidebar.selectbox('Véhiculé', ['Tout', 'Oui', 'Non'])


if gender_option == 'Tout':
    data_filtered = application_df.copy()
else:
    data_filtered = filter_df(application_df, 'CODE_GENDER', [gender_option])

if target_option == 'Accepté':
    data_filtered = filter_df(application_df, 'TARGET', [0])
elif target_option == 'Refusé':
    data_filtered = filter_df(application_df, 'TARGET', [1])

if car_option == 'Oui':
    data_filtered = filter_df(application_df, 'FLAG_OWN_CAR', ['Y'])
elif car_option == 'Non':
    data_filtered = filter_df(application_df, 'FLAG_OWN_CAR', ['N'])


bureau, bb, prev, pos, ins, cc = filter_data(data_filtered, bureau, bb, prev, pos, ins, cc)


# %%

st.header("Dashboard")

# %%
st.subheader('Applications')

data_filtered['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
# Some simple new features (percentages)
data_filtered['DAYS_EMPLOYED_PERC'] = data_filtered['DAYS_EMPLOYED'] / data_filtered['DAYS_BIRTH']
data_filtered['INCOME_CREDIT_PERC'] = data_filtered['AMT_INCOME_TOTAL'] / data_filtered['AMT_CREDIT']
data_filtered['INCOME_PER_PERSON'] = data_filtered['AMT_INCOME_TOTAL'] / data_filtered['CNT_FAM_MEMBERS']
data_filtered['ANNUITY_INCOME_PERC'] = data_filtered['AMT_ANNUITY'] / data_filtered['AMT_INCOME_TOTAL']
data_filtered['PAYMENT_RATE'] = data_filtered['AMT_ANNUITY'] / data_filtered['AMT_CREDIT']

new_features = ['DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC',
                'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']

ratio_target = data_filtered['TARGET'].mean()

st.subheader(f"Pourcentage de crédits refusés: {round(ratio_target * 100, 2)} %")

st.subheader('Applications')
st.write(data_filtered.head(20))

labels = data_filtered['NAME_CONTRACT_TYPE'].value_counts().index
values = data_filtered['NAME_CONTRACT_TYPE'].value_counts().values

#Contract type pie chart
fig1 = go.Figure(
    go.Pie(
    labels = labels,
    values = values,
    hoverinfo = "label+percent",
    textinfo = "value"
))

st.write("##### Type de contrat:")
st.plotly_chart(fig1)

# Amounts plot

x = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

y = data_filtered[x].mean(axis=0).values

fig2 = go.Figure(go.Bar(x=x, y=y, text=values))

fig2.update_layout(yaxis=dict(title='Moyenne ($)'))

st.write("##### Moyennes:")
st.plotly_chart(fig2)

# Rate plot plot

y = data_filtered[new_features].mean(axis=0).values


fig3 = go.Figure(go.Bar(x=new_features, y=y, text=values))

fig3.update_layout(yaxis=dict(title='Ratio', range=[0,1]))

st.write("##### Moyenne pourcentages:")
st.plotly_chart(fig3)

# %%

st.subheader('Bureau')
st.write(bureau.head())

st.subheader('Applications précédentes')
st.write(prev.head())

labels = prev['NAME_CONTRACT_STATUS'].value_counts().index
values = prev['NAME_CONTRACT_STATUS'].value_counts().values

#Contract type pie chart
fig = go.Figure(
    go.Pie(
    labels = labels,
    values = values,
    hoverinfo = "label+percent",
    textinfo = "value"
))

st.write("##### Status de contrat:")
st.plotly_chart(fig)

st.subheader('Acompte')
st.write(ins.head())

st.subheader('POS')
st.write(pos.head())

st.subheader('Solde de carte de crédits')
st.write(cc.head())




