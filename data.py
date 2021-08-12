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
    '''imports data and cleans for categorical analysis
       -drops lands and special (1 card) from rarity
       -changes price to str '''
    df = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
    #drop land cards and a special one
    df.drop(df.loc[df['rarity']=='\nBasic Land'].index, inplace=True)
    df.drop(df.loc[df['rarity']=='\nSpecial'].index, inplace=True)

    #change prices to strings
    df['market price ($)'] = df['market price ($)'].astype(str)
    df['foil price ($)'] = df['foil price ($)'].astype(str)
    df

    #change Nan to none in 'coloured mana' column to support colourless cards
    df['coloured mana'] = df['coloured mana'].fillna('none')

    return df

# %%
