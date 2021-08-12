#%%
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
    #drop land cards and a special one
df.drop(df.loc[df['rarity']=='\nBasic Land'].index, inplace=True)
df.drop(df.loc[df['rarity']=='\nSpecial'].index, inplace=True)

    #change prices to strings
df['market price ($)'] = df['market price ($)'].astype(str)
df['foil price ($)'] = df['foil price ($)'].astype(str)

X, y =df['rarity'], df['converted mana cost']

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





# Initialize the pipeline with any estimator    
pipe = Pipeline(steps=[('estimator', LogisticRegression())])

# list of dicts with estimator and estimator related parameters
params_grid = [{'estimator':[KNeighborsClassifier()],
               'estimator__algorithm': ['auto', 'ball_tree', 'brute'],
               'estimator__weights': ['uniform', 'distance'],
                },
    
                {
                'estimator':[SVC()],
                'estimator__C': [1, 10, 100, 1000],
                'estimator__gamma': [0.001, 0.0001],
                },
                {
                'estimator': [DecisionTreeClassifier()],
                'estimator__splitter': ['best', 'random'],
                'estimator__max_features': [None, "auto", "sqrt", "log2"],
                'estimator__criterion': ['gini', 'entropy']
                },
               
              ]

grid = GridSearchCV(pipe, params_grid)

start = time.time()
grid.fit(hot_enc_train, y_train)
end = time.time()
#print('fit time:', end-start)
val_score = grid.score(hot_enc_val, y_val)
test_score = grid.score(hot_enc_test, y_test)
print('val score:', val_score)
print('test score:', test_score)

#grid.fit(hot_enc_train, y_train)

val_score = grid.score(hot_enc_val, y_val)

test_score = grid.score(hot_enc_test, y_test)


scores_list = [end-start, val_score,test_score, grid.best_params_, grid.best_score_, grid.best_estimator_ ]
scores_list_df = pd.DataFrame(scores_list, ['fit time', 'val score', 'test score', 'best combination', 'best score', 'best all parameters'])
print(scores_list_df)


# %%

def plot_bar_graph():
    '''runs the models, gathers the data and plots the graph'''
    results = all_results()
    #plot a bargraph that show the different times and scores
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(6)]

    clf_names, val_score, training_time, val_time, test_score, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    val_time = np.array(val_time) / np.max(val_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices , training_time, .1, label="training time",
            color='c')
    plt.barh(indices + .2, val_score, .1, label="val score", color='navy')         
    plt.barh(indices + .4, val_time, .1, label="val time", color='darkorange')

    plt.barh(indices + .8, test_score, .1, label="test score", color='green')         
    plt.barh(indices + 1.0, test_time, .1, label="test time", color='lightgreen')



    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    #print the names in the y-axis of the graph
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


plot_bar_graph()






# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer=load_breast_cancer()
cancer_data   =cancer.data
cancer_target =cancer.target

x_train,x_test,y_train,y_test=train_test_split(cancer_data,cancer_target,test_size=0.2,random_state=2021)

param_grid_lr = {'C': [0.001,0.1,1,10],'penalty': ['l1','l2']}
gs_lr=GridSearchCV(LogisticRegression(solver='saga'),param_grid_lr)
x_train,x_test,y_train,y_test=train_test_split(cancer_data,
cancer_target,test_size=0.2,random_state=2021)
gs_lr.fit(x_train,y_train)
test_score=gs_lr.score(x_test,y_test)
print("test score:",test_score)
print("best combination: ",gs_lr.best_params_)
print("best score: ", gs_lr.best_score_)
print("best all parameters:",gs_lr.best_estimator_)
print("everything ",gs_lr.cv_results_)




# %%

