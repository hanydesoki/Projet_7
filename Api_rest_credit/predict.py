import pickle
from data_science.preprocessing import FillImputer, DfEncoderOneHot
from sklearn.base import BaseEstimator, TransformerMixin
from custom_preprocessing import CustomPreprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn import pipeline

with open('merger.pkl', 'rb') as f:
    merger = pickle.load(f)

with open('credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_score(application_df, bureau, bb, prev, pos, ins, cc):
    df = merger.transform(application_df, bureau, bb, prev, pos, ins, cc)

    try:
        df.drop('TARGET', axis=1, inplace=True)
    except KeyError:
        pass

    pred = model.predict_proba(df)[0][0]

    return {'score': pred}