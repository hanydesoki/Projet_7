import os

from data_science.data_management import DataGroup
from custom_preprocessing import filter_data

import streamlit as st
import pandas as pd

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

st.write("""
# Dashboard
Crédits changement""")