from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from pandas import set_option
from numpy import set_printoptions
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error,confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingClassifier, \
    VotingRegressor, StackingRegressor, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
import urllib
from sklearn.datasets import load_boston, load_breast_cancer, load_digits
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier,LogisticRegression
from sklearn.neighbors import NearestCentroid
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from mlxtend.evaluate import bias_variance_decomp
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.cluster import  DBSCAN
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB
data = load_digits()
X = data.data
Y = data.target
print(dir(data))
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,random_state=7)
model = LogisticRegression(max_iter=2000)
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
print(model.n_iter_)
prediction = model.predict(X_test)
print(confusion_matrix(Y_test,prediction))
plt.imshow(data.images[0])
plt.show()