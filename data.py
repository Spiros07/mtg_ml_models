#%%
import pandas as pd

def get_regr_data():
    '''imports data and cleans with respect to "market price ($)" - regression analysis'''
    df = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
    #drop cards that have a price > $200, only 14 cards in a sample of 9200 - outliers
    df.drop(df.loc[df['market price ($)'] > 200].index, inplace=True)
    df
    return df


def get_cat_data():
    '''imports data and cleans with respect to "rarity" - categorical analysis'''
    df = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
    #drop land cards and a special one
    df.drop(df.loc[df['rarity']=='\nBasic Land'].index, inplace=True)
    df.drop(df.loc[df['rarity']=='\nSpecial'].index, inplace=True)
    df
    return df




# %%
