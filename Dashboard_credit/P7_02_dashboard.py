import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# %%

decision = {0: 'Accepté', 1: 'Refusé'}
gender = {'M': 'Homme', 'F': 'Femme'}
car = {'Y': 'Oui', 'N': 'Non'}

st.set_page_config(layout="wide")

# Loading data

@st.cache(allow_output_mutation=True)
def load_data(folder):


    application_df = pd.read_csv(os.path.join(folder, 'application.csv'))
    bureau = pd.read_csv(os.path.join(folder, 'bureau.csv'))
    bb = pd.read_csv(os.path.join(folder, 'bureau_balance.csv'))
    prev = pd.read_csv(os.path.join(folder, 'previous_application.csv'))
    pos = pd.read_csv(os.path.join(folder, 'POS_CASH_balance.csv'))
    ins = pd.read_csv(os.path.join(folder, 'installments_payments.csv'))
    cc = pd.read_csv(os.path.join(folder, 'credit_card_balance.csv'))

    return application_df, bureau, bb, prev, pos, ins, cc

# %%

def filter_data(application_df, bureau, bb, prev, pos, ins, cc):
    skid_curr_filter = list(application_df['SK_ID_CURR'].unique())
    bureau_filtered = bureau[bureau['SK_ID_CURR'].isin(skid_curr_filter)]
    skid_bureau_filter = list(bureau_filtered['SK_ID_BUREAU'].unique())
    bb_filtered = bb[bb['SK_ID_BUREAU'].isin(skid_bureau_filter)]
    prev_filtered = prev[prev['SK_ID_CURR'].isin(skid_curr_filter)]
    pos_filtered = pos[pos['SK_ID_CURR'].isin(skid_curr_filter)]
    ins_filtered = ins[ins['SK_ID_CURR'].isin(skid_curr_filter)]
    cc_filtered = cc[cc['SK_ID_CURR'].isin(skid_curr_filter)]

    return bureau_filtered, bb_filtered, prev_filtered, pos_filtered, ins_filtered, cc_filtered

# %%

def comp_confmat(actual, predicted):

    # extract the different classes
    classes = [0, 1]

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat


# %%


application_df, bureau, bb, prev, pos, ins, cc = load_data('Dashboard_data')

application_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
# Some simple new features (percentages)
application_df['DAYS_EMPLOYED_PERC'] = application_df['DAYS_EMPLOYED'] / application_df['DAYS_BIRTH']
application_df['INCOME_CREDIT_PERC'] = application_df['AMT_INCOME_TOTAL'] / application_df['AMT_CREDIT']
application_df['INCOME_PER_PERSON'] = application_df['AMT_INCOME_TOTAL'] / application_df['CNT_FAM_MEMBERS']
application_df['ANNUITY_INCOME_PERC'] = application_df['AMT_ANNUITY'] / application_df['AMT_INCOME_TOTAL']
application_df['PAYMENT_RATE'] = application_df['AMT_ANNUITY'] / application_df['AMT_CREDIT']

new_features = ['DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC',
                'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']


# %%

def filter_df(df, col, classes):
    return df[df[col].isin(classes)]

# %%

col0 = st.columns(2) # Titre
col1 = st.columns(2) # Deux premiers graphs
col2 = st.columns(2) # Résumé client
col3 = st.columns(2) # Deux derniers graphs
col4 = st.columns(2) # Résumé client application précédentes

st.sidebar.header('Options')

gender_option = st.sidebar.selectbox('Genre', ['Tout'] + list(application_df['CODE_GENDER'].unique()))
target_option = st.sidebar.selectbox('Crédit', ['Tout', 'Accepté', 'Refusé'])
car_option = st.sidebar.selectbox('Véhiculé', ['Tout', 'Oui', 'Non'])
realty_option = st.sidebar.selectbox('Propriétaire', ['Tout', 'Oui', 'Non'])
client_id = st.sidebar.text_input('ID Client (SK_ID_CURR)')
treshold = st.sidebar.slider('Seuil de prédiction', value=50)

# check valid client id

try:
    client_filter = int(client_id) in application_df['SK_ID_CURR'].values
except Exception:
    client_filter = False

if not str(client_id).strip() == '' and not client_filter:
    col0[1].write("L'ID indiqué n'est pas dans la base de données")

if client_filter:
    client_id = int(client_id)
    col0[1].write(f"##### Comparaison avec le client {client_id}")


application_df['PRED'] = (application_df['SCORE']*100 < treshold).astype('int')


if gender_option == 'Tout':
    data_filtered = application_df.copy()
else:
    data_filtered = filter_df(application_df, 'CODE_GENDER', [gender_option])

if target_option == 'Accepté':
    data_filtered = filter_df(data_filtered, 'TARGET', [0])
elif target_option == 'Refusé':
    data_filtered = filter_df(data_filtered, 'TARGET', [1])

if car_option == 'Oui':
    data_filtered = filter_df(data_filtered, 'FLAG_OWN_CAR', ['Y'])
elif car_option == 'Non':
    data_filtered = filter_df(data_filtered, 'FLAG_OWN_CAR', ['N'])

if realty_option == 'Oui':
    data_filtered = filter_df(data_filtered, 'FLAG_OWN_REALTY', ['Y'])
elif realty_option == 'Non':
    data_filtered = filter_df(data_filtered, 'FLAG_OWN_REALTY', ['N'])



# Confusion matrix
conf_mat = comp_confmat(data_filtered['TARGET'], data_filtered['PRED'])

tn = conf_mat[0, 0]
tp = conf_mat[1, 1]
fn = conf_mat[0, 1]
fp = conf_mat[1, 0]

if client_filter:
    application_client = filter_df(application_df, 'SK_ID_CURR', [client_id])
    bureau_client, bb_client, prev_client, pos_client, ins_client, cc_client =\
        filter_data(application_client, bureau, bb, prev, pos, ins, cc)

    target_client = application_client['TARGET'].values[0]
    score_client = application_client['SCORE'].values[0]
    pred_client = application_client['PRED'].values[0]
    gender_client = gender[application_client['CODE_GENDER'].values[0]]
    car_client = car[application_client['FLAG_OWN_CAR'].values[0]]
    realty_client = car[application_client['FLAG_OWN_REALTY'].values[0]]

bureau, bb, prev, pos, ins, cc = filter_data(data_filtered, bureau, bb, prev, pos, ins, cc)




# %%

col0[0].header("Dashboard de Home Credit")


# %%
#st.subheader('Applications')

ratio_target = data_filtered['TARGET'].mean()

col0[0].write(f"Echantillon de {application_df.shape[0]} applications. ({data_filtered.shape[0]} après le filtre)")
col0[0].subheader(f"Pourcentage de crédits refusés: {round(ratio_target * 100, 2)} %")


#st.subheader('Applications')
#st.write(data_filtered.head(20))

labels = ['Vrai négatifs', 'Vrai positif', 'Faux négatif', 'Faux positif']
values = [tn, tp, fn, fp]
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

# Confusion matrix plot
fig1 = go.Figure(
    go.Pie(
    labels = labels,
    values = values,
    hoverinfo = "label+percent",
    textinfo = "value",
    marker=dict(colors=colors))
)

fig1.update_layout(
    autosize=False,
    width=400,
    height=400,)

col1[0].write("##### Résultat de prédiction du model")
col1[0].plotly_chart(fig1)

if client_filter:
    col2[0].write(f"""######
             -Décision réelle: {decision[target_client]}
             -Score du client: {score_client}
             -Prédiction du modèle: {decision[pred_client]}""")

    col2[1].write(f"""######
             -Genre: {gender_client}
             -Véhiculé: {car_client}
             -Propriétaire: {realty_client}""")



# Amounts plot

x = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

y_0 = data_filtered[data_filtered['TARGET'] == 0][x].mean(axis=0).values
y_1 = data_filtered[data_filtered['TARGET'] == 1][x].mean(axis=0).values

amount_values_0 = go.Bar(x=x, y=y_0,showlegend=True, name='Accepté')
amount_values_1 = go.Bar(x=x, y=y_1,showlegend=True, name='Refusé')
#trace1 = go.Bar(x=x, y=[0],showlegend=False,hoverinfo='none')
#trace2 = go.Bar(x=x, y=[0], yaxis='y2',showlegend=False,hoverinfo='none')

data = [amount_values_0,amount_values_1]


if client_filter:
    y_client = application_client[x].mean(axis=0).values
    amout_values_client =  go.Bar(x=x, y=y_client, showlegend=True,
                                  name=f'Client ({client_id})')

    data.append(amout_values_client)

layout = go.Layout(barmode='group',
                   legend=dict(x=0.7, y=1.2,orientation="h"),
                   yaxis=dict(title='Moyennes ($)'),
                   yaxis2=dict(title = '',
                               overlaying = 'y',
                               side='right'))

#fig2.update_layout(yaxis=dict(title='Moyenne ($)'))

fig2 = go.Figure(data=data, layout=layout)

fig2.update_layout(
    autosize=False,
    width=500,
    height=400,)

col1[1].write("##### Prix, revenus, montants:")
col1[1].plotly_chart(fig2)

# Rate plot plot

y_0 = data_filtered[data_filtered['TARGET'] == 0][new_features].mean(axis=0).values
y_1 = data_filtered[data_filtered['TARGET'] == 1][new_features].mean(axis=0).values

perc_values_0 = go.Bar(x=new_features, y=y_0,showlegend=True, name='Accepté')
perc_values_1 = go.Bar(x=new_features, y=y_1,showlegend=True, name='Refusé')
#trace1 = go.Bar(x=x, y=[0],showlegend=False,hoverinfo='none')
#trace2 = go.Bar(x=x, y=[0], yaxis='y2',showlegend=False,hoverinfo='none')

data = [perc_values_0,perc_values_1]

if client_filter:
    y_client = application_client[new_features].mean(axis=0).values
    amout_values_client =  go.Bar(x=new_features, y=y_client, showlegend=True,
                                  name=f'Client ({client_id})')

    data.append(amout_values_client)

layout = go.Layout(barmode='group',
                   legend=dict(x=0.7, y=1.2,orientation="h"),
                   yaxis=dict(title='Ratio'),
                   yaxis2=dict(title = '',
                               overlaying = 'y',
                               side='right'))

#fig2.update_layout(yaxis=dict(title='Moyenne ($)'))

fig3 = go.Figure(data=data, layout=layout)

fig3.update_layout(
    autosize=False,
    width=500,
    height=400,)

col3[0].write("##### Rapports de montants:")
col3[0].plotly_chart(fig3)

# %%

#st.subheader('Bureau')
#st.write(bureau.head())

#st.write(prev.head())

labels = prev['NAME_CONTRACT_STATUS'].value_counts().index
values = prev['NAME_CONTRACT_STATUS'].value_counts().values

if client_id:
    col4[1].write(f'Application(s) précédentes du client {client_id}:')
    if prev_client.shape[0] > 0:
        labels_client = prev_client['NAME_CONTRACT_STATUS'].value_counts().index
        values_client = prev_client['NAME_CONTRACT_STATUS'].value_counts().values
        for lab, val in zip(labels_client, values_client):
            col4[1].write(f'-{lab}: {val}')
    else:
        col4[1].write("Pas d'application enregistrée dans le passé")

# Contract type pie chart
fig4 = go.Figure(
    go.Pie(
    labels = labels,
    values = values,
    hoverinfo = "label+percent",
    textinfo = "value"
))

fig4.update_layout(
    autosize=False,
    width=400,
    height=400,)

col3[1].write("##### Status des contrats précédents:")
col3[1].plotly_chart(fig4)

#st.subheader('Acompte')
#st.write(ins.head())

#st.subheader('POS')
#st.write(pos.head())

#st.subheader('Solde de carte de crédits')
#st.write(cc.head())




