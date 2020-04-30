# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:15:29 2020

@author: enriq
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_ind
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()

label = "class"
train = pd.DataFrame(iris.data, columns = iris.feature_names)  # we only take the first two features.
y = pd.DataFrame(iris.target, columns = ["class"])

class MultiColumnDropper():
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        '''
        Drops columns of X specified in self.columns 
        If no columns specified, returns X
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                try:
                    output = output.drop(col, axis = 1)
                except:
                    pass
        else:
            pass
        
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# "---------------------------------------------------------"
# "--------------------UNIENDO DATAFRAMES ------------------"
# "---------------------------------------------------------"
# Dividiendo columnas por tipo

label_cols = [label]

numeric_cols = train.dtypes[train.dtypes == np.float64].index.to_list() +\
    train.dtypes[train.dtypes == np.int64].index.to_list()
numeric_cols = [col for col in numeric_cols if col not in label_cols]

cat_cols = train.dtypes[(train.dtypes == "O") | (train.dtypes == "category")].index[1:].to_list()

# "---------------------------------------------------------"

# Dejar una observación por cliente-mes
train_temp = train

# Feature Selection 

# Eliminando columnas con nulos

droped_null_cols = []
for col in (cat_cols + numeric_cols):
    if train_temp[col].isna().sum()/len(train_temp[col]) > 0.40:
        # print(str(train_temp[col].isna().sum()/len(train_temp[col])), col)
        droped_null_cols.append(col)

# Análisis Columnas numéricas
train_num = train_temp[numeric_cols]

# Eliminando una de cada dos columnas numéricas correlacionadas por ser redundantes

corrmat = train_num.corr()
droped_corr_cols = []

for col in corrmat.columns:
    if col not in droped_corr_cols:
        a = corrmat[col][(abs(corrmat[col]) > 0.9) & (abs(corrmat[col]) < 1)]
        for colname in a.index:
            if colname in droped_corr_cols:
                pass
            else:
                # print(colname + " se correlaciona mucho con " + col +
                #       "con un PCC de " + str(corrmat.loc[colname, col]) + 
                #       "\n por lo tanto nos deshacemos de ella por aportar\n la misma información")
                droped_corr_cols.append(colname)
                corrmat = corrmat.drop(colname, axis = 1)

# Graficando histogramas de columnas numéricas

# for col in train_num.columns:
#      train_num[col].plot.hist(title = col)
#      s = train_num.describe()[col].to_string() + \
#          "\nMissing Values: " + str(train_num.isnull().sum()[col]) + \
#          "\nMissing Values %: " + str(round(train_num.isnull().sum()[col]/len(train_num),4))
#      plt.figtext(1, 0.5, s)
#      plt.show()

droped_ttest_cols = []         
# * Evaluar normalidad "skewness"
target = y[label]
t_sel = [0] * len(train_num.columns) # señala qué variables pueden ayudar a predecir target
t_ctr = 0 # contador

for col in train_num.columns:
    # Shapiro-Wilk test
    stat, p = shapiro(train_num[col])
    #print('Statistics={:.3f}, p={:.3f}'.format(stat, p))
    
    if p > 0.05: # no se rechaza la H0 según la cual la distribución de estos datos es similar a la gaussiana
        # t-test
        # print(col)
        # separación de datos según la aceptación del crédito
        t0 = train_num[col][target == 0]
        t1 = train_num[col][target == 1]
        stat, p = ttest_ind(t0, t1, nan_policy = "omit", equal_var = False)
        # print('T-statistic={:.3f}, p={:.3f}'.format(stat, p))
        
        if p < 0.05: # se rechaza la H0 según la cual las medias de t0 y t1 no difieren significativamente
            t_sel[t_ctr] = 1
        else:
            droped_ttest_cols.append(col)
            pass
    t_ctr += 1
    
t_selec = pd.DataFrame(t_sel, index = train_num.columns)


droped_cols = []

final_num_cols = [col for col in numeric_cols if col not in droped_cols]
final_cat_cols = [col for col in cat_cols if col not in droped_cols]

feature_selector = Pipeline(steps = [
    ("null_dropper", MultiColumnDropper(droped_cols))])

# Imputer categóricas con más frecuente
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown = "ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("feature_select", feature_selector, droped_cols),
        ("cat", categorical_transformer, final_cat_cols),
        ('num', numeric_transformer, final_num_cols)])

# pd.DataFrame(preprocessor.fit_transform(x_train))

param_grid_lr = {
    'classifier__C': [0.01, 0.001, 0.1, 1.0],
}

param_grid_rf = {
    'classifier__n_estimators': [150, 200]
}

lr_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter = 500))])    


rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_jobs = -1))])    

grid_search_lr = GridSearchCV(lr_clf, param_grid_lr, cv = 5, n_jobs = -1, scoring = "accuracy")
grid_search_rf = GridSearchCV(rf_clf, param_grid_rf, cv = 5, n_jobs = -1, scoring = "accuracy")

features = [col for col in train_temp.columns if col != label]
x_train = train_temp[features]
y_train = y

grid_search_rf.fit(x_train, y_train)
grid_search_lr.fit(x_train, y_train)

#-----------------------------------------

# evaluator
evaluation_df_rf = pd.DataFrame(grid_search_rf.cv_results_)
evaluation_df_lr = pd.DataFrame(grid_search_lr.cv_results_)
print("Score RF: ", grid_search_rf.best_score_)
print("Score LR: ", grid_search_lr.best_score_)

pp = grid_search_rf.best_estimator_['preprocessor']
pp_cols = get_column_names_from_ColumnTransformer(pp)
imp = grid_search_rf.best_estimator_['classifier'].feature_importances_
index = np.argsort(imp)
index = np.concatenate((index[:15], index[15:16]))

plt.title('Feature importances')
plt.barh(range(len(index)), imp[index], color='b', align='center')
plt.yticks(range(len(index)), [pp_cols[i] for i in index])
plt.xlabel('Relative Importance')
plt.figure(figsize = (12, 12))
plt.savefig('plt_' + str(month.date()))

# Segundo fit con variables importantes

    

