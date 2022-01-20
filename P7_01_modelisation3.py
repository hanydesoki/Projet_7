# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:32:22 2021

@author: Hany
"""
import os
import gc
import copy
import pickle
from data_science import prexplo, explo
from data_science.utils import ContextTimer
from data_science.data_management import DataGroup
from data_science.evaluations import evaluate_class
from data_science.preprocessing import FillImputer
from data_science.prexplo import describe_columns
from data_science.explo import MultiDimVizualisation
from custom_preprocessing import CustomPreprocessing, filter_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import fbeta_score, make_scorer, f1_score, recall_score, accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn import pipeline


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

ContextTimer.clear_timers()

# %%

print(xgboost.__version__)
# %%

with ContextTimer('Loading all datas'):
    DATA_FOLDER = 'Datas'
    all_datas = DataGroup(DATA_FOLDER)
    all_datas

# %%


SAMPLE_DATA = True

N_SAMPLE = 10000

with ContextTimer('Splitting data'):
    application_df = all_datas['application_train']
    bureau = all_datas['bureau']
    bb = all_datas['bureau_balance']
    prev = all_datas['previous_application']
    pos = all_datas['POS_CASH_balance']
    ins = all_datas['installments_payments']
    cc = all_datas['credit_card_balance']

    if SAMPLE_DATA:
        application_df = application_df.sample(N_SAMPLE, random_state=0)
    application_df = application_df[application_df['CODE_GENDER'] != 'XNA']
    application_train, application_test = train_test_split(application_df,
                                                           test_size=0.2,
                                                           random_state=0)
    print('\nTrain:\n')
    print(application_train['TARGET'].value_counts())
    print('\nTest:\n')
    print(application_test['TARGET'].value_counts())

# %%

# Filter data to match application

bureau_train, bb_train, prev_train, pos_train, ins_train, cc_train =\
    filter_data(application_train,
            bureau, bb, prev, pos, ins, cc)

bureau_test, bb_test, prev_test, pos_test, ins_test, cc_test =\
    filter_data(application_test,
                bureau, bb, prev, pos, ins, cc)

# %%

with ContextTimer('Merging and preprocessing all datas'):
    with ContextTimer('Fit merger'):
        merger = CustomPreprocessing(crit_missing_rate=0.6)

        merger.fit(application_train,
                                     bureau_train,
                                     bb_train,
                                     prev_train,
                                     pos_train,
                                     ins_train,
                                     cc_train)
    with ContextTimer('Transform train'):
        df_train = merger.transform(application_train,
                                     bureau_train,
                                     bb_train,
                                     prev_train,
                                     pos_train,
                                     ins_train,
                                     cc_train)
    with ContextTimer('Transform test'):
        df_test = merger.transform(application_test,
                                     bureau_test,
                                     bb_test,
                                     prev_test,
                                     pos_test,
                                     ins_test,
                                     cc_test)


# %%

X_train = df_train.drop('TARGET', axis=1)
y_train = df_train['TARGET']
X_test = df_test.drop('TARGET', axis=1)
y_test = df_test['TARGET']

# %%

vizualization = MultiDimVizualisation(transformer=Pipeline(steps=[('imputer', FillImputer()),
                                                                   ('scaler', StandardScaler())]))

vizualization.fit_transform_plot(X_train, y_train)
# %%

beta = 9

fbeta_metrics = make_scorer(fbeta_score, greater_is_better=True, beta=beta)


# %%

with ContextTimer('Modelisation'):

    # imblearn pipeline
    base_model = pipeline.Pipeline(steps=[('imputer', FillImputer()),
                                          ('scaler', StandardScaler()),
                                          ('pca', PCA(n_components=0.95)),
                                          ('smt', SMOTE(random_state=0)),
                                          ('xgb', XGBClassifier(use_label_encoder=False,
                                                                n_estimators=50))])

    with ContextTimer('Parameter_optimizaion'):

        param_grid={
                    'xgb__reg_lambda':[2, 1],
                    'xgb__gamma':[0, 0.2, 0.1]
                    ,'xgb__eta':[0.05, 0.04]
                    ,"xgb__max_depth":[3, 5]}

        grid = GridSearchCV(copy.deepcopy(base_model), param_grid=param_grid, cv=3,
                            scoring=fbeta_metrics, verbose=3)

        grid.fit(X_train, y_train)

        best_params = grid.best_params_

        print('\n Best parameters:', best_params)

# %%

with ContextTimer('Fit with best model'):
    final_model = pipeline.Pipeline(steps=[('imputer', FillImputer()),
                                          ('scaler', StandardScaler()),
                                          ('smt', SMOTE(random_state=0)),
                                          ('xgb', XGBClassifier(
                                            n_estimators=50))]).set_params(**best_params)


    final_model.fit(X_train, y_train)

    evaluate_class(final_model, X_test, y_test)


# %%

with ContextTimer('Save models'):
    with open('merger.pkl', 'wb') as f1:
        pickle.dump(merger, f1)
    with open('credit_model.pkl', 'wb') as f2:
        pickle.dump(final_model, f2)

# %%

feat_imp = pd.DataFrame(final_model.named_steps['xgb'].feature_importances_, index=X_train.columns)

# %%

# Getting score of all application_df for the dashboard

bureau, bb, prev, pos, ins, cc =\
    filter_data(application_df,
            bureau, bb, prev, pos, ins, cc)

X = merger.transform(application_df,
                     bureau,
                     bb,
                     prev,
                     pos,
                     ins,
                     cc)

X.drop('TARGET', axis=1, inplace=True)
# %%

scores = final_model.predict_proba(X)[:,0]

application_df['SCORE'] = scores

# %%

directory = 'Dashboard_data'

if not os.path.exists(directory):
    os.makedirs(directory)

# save data

application_df.to_csv(os.path.join(directory, 'application.csv'))
bureau.to_csv(os.path.join(directory, 'bureau.csv'))
bb.to_csv(os.path.join(directory, 'bureau_balance.csv'))
prev.to_csv(os.path.join(directory, 'previous_application.csv'))
pos.to_csv(os.path.join(directory, 'POS_CASH_balance.csv'))
ins.to_csv(os.path.join(directory, 'installments_payments.csv'))
cc.to_csv(os.path.join(directory, 'credit_card_balance.csv'))

# %%

# Plot scores with tresholds

y_true = application_df['TARGET'].values

acc = []
rec = []
pre = []
f1 = []
f9 = []

tresholds = np.linspace(0, 1, 100)

for tresh in tresholds:
    y_pred = (scores < tresh).astype('int')

    acc.append(accuracy_score(y_true, y_pred))
    rec.append(recall_score(y_true, y_pred))
    pre.append(precision_score(y_true, y_pred))
    f1.append(f1_score(y_true, y_pred))
    f9.append(fbeta_score(y_true, y_pred, beta=9))


plt.figure()

plt.plot(tresholds, acc, label='accuracy')
plt.plot(tresholds, rec, label='recall')
plt.plot(tresholds, pre, label='precision')
plt.plot(tresholds, f1, label='f1')
plt.plot(tresholds, f9, label='f9')

plt.legend()
plt.xlabel('Treshold')
plt.ylabel('Score')
plt.grid(True)

plt.savefig('treshold_score.jpeg')
