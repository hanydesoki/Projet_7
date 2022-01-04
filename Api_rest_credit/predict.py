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

def predict():
    pass