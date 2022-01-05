from predict import predict_score
from data_science.data_management import DataGroup

# %%

TARGET = 0
i = 0

path = f'Examples/Example_{TARGET}_{i}'

all_data = DataGroup(path)

application_df = all_data['application']
bureau = all_data['bureau']
bb = all_data['bureau_balance']
prev = all_data['previous_application']
pos = all_data['POS_CASH_balance']
ins = all_data['instalment_payments']
cc = all_data['credit_card_balance']


# %%

print(predict_score(application_df, bureau, bb, prev, pos, ins, cc))