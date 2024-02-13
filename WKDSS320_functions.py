# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:07:25 2023

@author: Hiren Patel
"""
#*****************************************************************************
# Functions for use in the notebooks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def RandF_quick_analysis(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn import metrics
    print("ACCURACY OF THE MODEL: ", round(metrics.accuracy_score(y_test, y_pred),2)*100, "%")
    return clf

# Function derived from https://vitalflux.com/k-means-elbow-point-method-sse-inertia-plot-python/
def drawSSEPlotForKMeans(df, column_indices, n_clusters=8, max_iter=50, tol=1e-04, init='k-means++', n_init=10, algorithm='auto'):
    from sklearn.cluster import KMeans
    inertia_values = []
    for i in range(1, n_clusters+1):
        km = KMeans(n_clusters=i+1, max_iter=max_iter, tol=tol, init=init, n_init=n_init, random_state=1, algorithm=algorithm)
        km.fit_predict(df.iloc[:, column_indices])
        inertia_values.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, n_clusters+1), inertia_values, '-ro')
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('SSE / Inertia', fontsize=15)
    plt.title('SSE / Inertia vs No. Of Clusters', fontsize=15)
    plt.grid()
    return ax

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, X, y):
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    fig, ax = plt.subplots(figsize=(7,7))
    # title for the plots
    title = ('Decision surface')

    #Reshape data
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

#    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # plot the training data
    axHandle = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Variable 1')
    ax.set_xlabel('Variable 0')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend(*axHandle.legend_elements())
    plt.show()
    
    return ax

