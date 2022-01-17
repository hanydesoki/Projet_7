import os
import flask
from flask import Flask, jsonify, request
import json
from predict import predict_score
import pandas as pd

TARGET = 0
i = 0

path = f'../Examples/Example_{TARGET}_{i}'


app = Flask(__name__)\

@app.route('/predict', methods=['GET'])
def predict():

    # Request json file
    all_data = request.get_data(as_text=True)

    all_data = json.loads(all_data)

    # Sparse data
    application_df = pd.DataFrame(all_data['application.csv'])
    bureau = pd.DataFrame(all_data['bureau.csv'])
    bb = pd.DataFrame(all_data['bureau_balance.csv'])
    prev = pd.DataFrame(all_data['previous_application.csv'])
    pos = pd.DataFrame(all_data['POS_CASH_balance.csv'])
    ins = pd.DataFrame(all_data['instalment_payments.csv'])
    cc = pd.DataFrame(all_data['credit_card_balance.csv'])

    SK_ID_CURR = application_df.iloc[0, 0]

    # Security if empty DataFrame
    if len(bureau) == 0:
        bureau['SK_ID_CURR'] = [SK_ID_CURR]
        if len(bb) != 0:
            SK_ID_BUREAU = bb.iloc[0, 1]
        else:
            SK_ID_BUREAU = 0
        bureau['SK_ID_BUREAU'] = SK_ID_BUREAU

    if len(prev) == 0:
        prev['SK_ID_CURR'] = [SK_ID_CURR]
        prev['SK_ID_PREV'] = 0



    # Predict score
    pred = predict_score(application_df, bureau, bb, prev, pos, ins, cc)

    response = json.dumps({"score": str(pred)})

    return response, 200

if __name__ == '__main__':
     app.run(debug=True)
