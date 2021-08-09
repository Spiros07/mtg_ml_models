#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from data import get_cat_data



dtf = get_cat_data()

#choose columns for classification - 'market price ($)', 'main type', 'rarity', 'market price ($)', 
#            'foil price ($)', 'converted mana cost'. 
X, y =dtf['rarity'], dtf['market price ($)']

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)   
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=20)


X_train_df = pd.DataFrame(X_train)
X_val_df = pd.DataFrame(X_val)
X_test_df = pd.DataFrame(X_test)


X_train_matrix = X_train_df.values.reshape(-1,1)
X_val_matrix = X_val_df.values.reshape(-1,1)
X_test_matrix = X_test_df.values.reshape(-1,1)

#use one hot encode
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train_matrix)

hot_enc_train = enc.transform(X_train_matrix).toarray()
hot_enc_val = enc.transform(X_val_matrix).toarray()
hot_enc_test = enc.transform(X_test_matrix).toarray()


# %%
