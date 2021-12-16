import os

from data_science.data_management import DataGroup
from custom_preprocessing import filter_data

import streamlit as st
import pandas as pd

# %%

# Loading data
SAMPLE_DATA = True

N_SAMPLE = 10000

FOLDER = 'Datas'

all_datas = DataGroup(FOLDER)

application_df = all_datas['application_train']
bureau = all_datas['bureau']
bb = all_datas['bureau_balance']
prev = all_datas['previous_application']
pos = all_datas['POS_CASH_balance']
ins = all_datas['installments_payments']
cc = all_datas['credit_card_balance']

application_df = application_df[application_df['CODE_GENDER'] != 'XNA']

if SAMPLE_DATA:
    application_df = application_df.sample(N_SAMPLE, random_state=0)

# Filter data to match application

bureau, bb, prev, pos, ins, cc = filter_data(application_df,

                                                         bureau, bb, prev, pos, ins, cc)
# %%
st.write("""
# Dashboard
Crédits""")

# %%

cmd = "streamlit run C:/Users/Hany/Desktop/Data_Scientist_OpenClassrooms/Projet_7/P7_02_dashboard.py"

os.system(cmd)