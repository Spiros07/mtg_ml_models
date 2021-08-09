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

#choose columns for classification - 'market price ($)', 'main type', 'rarity', 'market price ($)', 
#            'foil price ($)', 'converted mana cost'. 
X, y =dtf['rarity'], dtf['converted mana cost']

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


#the columns used for analysis
target_names = ['rarity', 'market price ($)']


def benchmark(clf):
    '''-fits the data and predicts for val and test set
       -calculates the time needed for the model to run'''
       
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(hot_enc_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(hot_enc_val)
    val_time = time() - t0
    print("val time:  %0.3fs" % val_time)

    score = metrics.accuracy_score(y_val, pred)
    print("accuracy:   %0.3f" % score)

    
    #confusion matrix if needed
    # print("confusion matrix:")
    # print(metrics.confusion_matrix(y_val, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, val_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))


for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3, max_iter=5000)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=150,
                                           penalty=penalty)))



# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3, max_iter=5000))),
  ('classification', LinearSVC(penalty="l2"))])))



# %%
