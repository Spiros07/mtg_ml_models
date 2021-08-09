import pandas as pd

def get_data():
    '''imports data and cleans with respect to "market price ($)" - regression analysis'''
    df = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
    
    return df

