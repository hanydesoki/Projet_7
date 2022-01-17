import os
import json

from data_science.data_management import DataGroup
from custom_preprocessing import filter_data

import pandas as pd

# %%
DATA_PATH = 'Datas'

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

N_SAMPLE = 3

application_0 = application_df[application_df['TARGET'] == 0].sample(N_SAMPLE)
application_1 = application_df[application_df['TARGET'] == 1].sample(N_SAMPLE)


# %%

def save_data(app, bureau, bb, prev, pos, ins, cc, folder):

    data = {}
    data['application.csv'] = app.to_dict()
    data['bureau.csv'] = bureau.to_dict()
    data['bureau_balance.csv'] = bb.to_dict()
    data['previous_application.csv'] = prev.to_dict()
    data['POS_CASH_balance.csv'] = pos.to_dict()
    data['instalment_payments.csv'] = ins.to_dict()
    data['credit_card_balance.csv'] = cc.to_dict()

    #return data

    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f)




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

        test = save_data(app_filtered, bureau_filtered, bb_filtered, prev_filtered,
                  pos_filtered, ins_filtered, cc_filtered, path)
