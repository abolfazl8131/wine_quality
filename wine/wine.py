from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from pandas import set_option
from numpy import set_printoptions
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, confusion_matrix, \
    precision_score
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso, LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingClassifier, \
    VotingRegressor, StackingRegressor, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
import urllib
from sklearn.datasets import load_boston, load_breast_cancer
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
#loading data
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.shape)
#corr
corr = df.corr(method='pearson')

# X Y
X = df.iloc[:,:11]
Y = df.iloc[:,11]
#pca
#pca = PCA()
#X = pca.fit_transform(X)
#training set / test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8, test_size=0.2,random_state=7)

#scaling
scale = StandardScaler()
X_train= scale.fit_transform(X_train)

#model fitting
estimators = [('LR',LogisticRegressionCV(max_iter=2000)),
               ('RFC',RandomForestClassifier(n_estimators=50)),
               ('DRC',DecisionTreeClassifier()),]
model = VotingClassifier(estimators,voting='hard')



model.fit(X_train,Y_train)
X_test = scale.transform(X_test)
print('model_score:',model.score(X_test,Y_test))
prediction = model.predict(X_test)
acc = accuracy_score(Y_test,prediction)
per = precision_score(Y_test,prediction,average='macro')
print('per:',per)
print('accuracy:',acc)
print(prediction)
print(Y_test)
print(confusion_matrix(Y_test,prediction))
print(df['quality'].value_counts())
model_name = 'red_wine_model.sav'
sclaer = 'scaler.sav'
pickle.dump(model, open(model_name,'wb'))
pickle.dump(scale, open(sclaer,'wb'))
