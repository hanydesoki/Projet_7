import pandas as pd

def describe_columns(df):
    '''Get for each columns: the type, the number and the frequence of NaN and the number of unique values

            Parameters:
                    df (DataFrame): A DataFrame

            Returns:
                    desc_df (DataFrame): A DataFrame with described columns
    '''
    desc_df = {'Type':[],
              'NaN count':[],
              'NaN frequency':[],
              'Number of unique values':[]}
    for col in df.columns:
        desc_df['Type'].append(df[col].dtype)
        desc_df['NaN count'].append(pd.isnull(df[col]).sum())
        desc_df['NaN frequency'].append(pd.isnull(df[col]).mean())
        desc_df['Number of unique values'].append(len(df[col].unique()))
    return pd.DataFrame(desc_df, index=df.columns)
