import pickle


with open('models/merger.pkl', 'rb') as f:
    merger = pickle.load(f)

with open('models/credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_score(application_df, bureau, bb, prev, pos, ins, cc):
    df = merger.transform(application_df, bureau, bb, prev, pos, ins, cc)

    try:
        df.drop('TARGET', axis=1, inplace=True)
    except KeyError:
        pass


    pred = model.predict_proba(df)[0][0]

    return pred