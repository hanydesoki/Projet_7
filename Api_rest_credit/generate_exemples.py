import os

from data_science.data_management import DataGroup
from custom_preprocessing import filter_data

import pandas as pd

# %%
DATA_PATH = '../Datas'

all_data = DataGroup(DATA_PATH)

# %%

application_df = all_data['application_train']
bureau = all_data['bureau']
bb = all_data['bureau_balance']
prev = all_data['previous_application']
pos = all_data['POS_CASH_balance']
ins = all_data['installments_payments']
cc = all_data['credit_card_balance']

# %%

N_SAMPLE = 2

application_0 = application_df[application_df['TARGET'] == 0].sample(N_SAMPLE)
application_1 = application_df[application_df['TARGET'] == 1].sample(N_SAMPLE)


# %%

def save_data(app, bureau, bb, prev, pos, ins, cc, folder):
    app.to_csv(os.path.join(folder, 'application.csv'))
    bureau.to_csv(os.path.join(folder, 'bureau.csv'))
    bb.to_csv(os.path.join(folder, 'bureau_balance.csv'))
    prev.to_csv(os.path.join(folder, 'previous_application.csv'))
    pos.to_csv(os.path.join(folder, 'POS_CASH_balance.csv'))
    ins.to_csv(os.path.join(folder, 'instalment_payments.csv'))
    cc.to_csv(os.path.join(folder, 'credit_card_balance.csv'))

# %%


for target, application in enumerate([application_0, application_1]):
    for i in range(N_SAMPLE):

        app_filtered = pd.DataFrame(application.iloc[i]).T
        bureau_filtered, bb_filtered, prev_filtered, pos_filtered,\
        ins_filtered, cc_filtered =\
                filter_data(app_filtered, bureau, bb, prev, pos, ins, cc)

        path = f'Examples/Example_{target}_{i}'

        if not os.path.exists(path):
            os.makedirs(path)

        save_data(app_filtered, bureau_filtered, bb_filtered, prev_filtered,
                  pos_filtered, ins_filtered, cc_filtered, path)
