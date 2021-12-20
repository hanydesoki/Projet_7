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

lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus.\
    Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed,\
        dolor. Cras elementum ultrices diam.\
            Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue. Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim. Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh. Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam sodales hendrerit."

ratio_target = application_df['TARGET'].mean()

st.header("Dashboard")
st.subheader(f"Pourcentage de crédits refusés: {ratio_target*100} %")

st.subheader('Applications')

st.write(application_df)

