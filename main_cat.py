#%%
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from data import get_cat_data





dtf = get_cat_data()

#choose any 2 columns for classification -  'main type', 
#                                           'rarity', 'market price ($)', 
#                                           'foil price ($)', 'converted mana cost'. 
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


def benchmark(clf):
    '''-fits the data and predicts for val and test set
       -calculates the time needed for the model to run'''
          
    t0 = time()
    clf.fit(hot_enc_train, y_train)
    train_time = time() - t0

    t0 = time()
    val_pred = clf.predict(hot_enc_val)
    val_time = time() - t0

    val_score = metrics.accuracy_score(y_val, val_pred)

    t0 = time()
    test_pred = clf.predict(hot_enc_test)
    test_time = time() - t0

    test_score = metrics.accuracy_score(y_test, test_pred)

    #confusion matrix if needed
    # print("confusion matrix:")
    # print(metrics.confusion_matrix(y_val, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, val_score, train_time, val_time, test_score, test_time



def all_results():
    results = []

    results.append(benchmark(Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                    tol=1e-3, max_iter=6000))),
    ('classification', LinearSVC(penalty="l2"))])))

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
            (Perceptron(max_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(max_iter=50),
            "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(), "Random forest")):

        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                        tol=1e-3, max_iter=6000)))

        results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=150,
                                            penalty=penalty)))


    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                        penalty="elasticnet")))

    # Train NearestCentroid without threshold
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))
    results.append(benchmark(ComplementNB(alpha=.1)))
  

    return results

results = all_results()
results_df = pd.DataFrame(results, columns=['Name', 'Train Time', 'Val Time', 'Val Acc', 'Test Time', 'Test Acc'])
results_df.head(15)

# %%

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

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()






# %%
