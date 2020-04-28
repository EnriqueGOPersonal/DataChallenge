# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:57:38 2020

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
        print("argumento pasado a transformer")
        print(self.columns)
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

test = pd.read_csv(r"./data/Base1_test.csv")

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')

# Carga datos
base_1 = pd.read_csv(r"./data/Base1_train.csv", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Base cotizaciones
base_2 = pd.read_csv(r"./data/Base2.csv", sep = ";", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Información sociodemográfica + digital
base_3 = pd.read_csv(r"./data/Base3.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base productos BBVA
base_4 = pd.read_csv(r"./data/Base4.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de saldos en el Sistema Financiero
base_5 = pd.read_csv(r"./data/Base5.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de consumos con tarjeta

for col in ["USO_BI_M0", "USO_BI_M1", "USO_BI_M2", "USO_BM_M0", "USO_BM_M1", "USO_BM_M2"]:
    base_2[col] = base_2[col].astype(str)

base_4["ST_CREDITO"] = base_4["ST_CREDITO"].astype(str)
base_5[["CD_DIVISA", "TP_TARJETA"]] = base_5[["CD_DIVISA", "TP_TARJETA"]].astype(str)

# Depurando Trainset 

base_1 = base_1.groupby(['MES_COTIZACION', 'COD_CLIENTE', 'GARANTIA', 'IMPORTE',
       'PLAZO', 'TEA_MINIMA'], as_index = False).agg({"FLG_DESEMBOLSO": "max"})

base_1 = base_1.sort_values(["MES_COTIZACION", "COD_CLIENTE"], ascending = False)\
    .drop_duplicates(["MES_COTIZACION", "COD_CLIENTE"], keep = "first")

test = test.drop_duplicates(['MES_COTIZACION', 'COD_CLIENTE', 'GARANTIA', 'IMPORTE',
       'PLAZO', 'TEA_MINIMA'])\
    .drop_duplicates(["COD_CLIENTE", "MES_COTIZACION"])

# Exploratory Data Analysis
## Base_1

#Cantidad de clientes unicos / longitud del df
len(base_1.COD_CLIENTE.unique())/ base_1.shape[0]
# PLT no son únicos los clientes

## Base_2
#Cantidad de clientes unicos / longitud del df
len(base_2.COD_CLIENTE.unique())/base_2.shape[0]
# PLT no son únicos los clientes

## Base_3
#Cantidad de clientes unicos / longitud del df
len(base_3.COD_CLIENTE.unique())/base_3.shape[0]
# PLT no son únicos los clientes

## Base_4
#Cantidad de clientes unicos / longitud del df
len(base_4.COD_CLIENTE.unique())/base_4.shape[0]
# PLT no son únicos los clientes

## Base_5
# Cantidad de clientes unicos / longitud del df
len(base_5.COD_CLIENTE.unique())/base_5.shape[0]
# PLT no son únicos los clientes

# En ningún df hay codigos de clientes nulos
base_1.COD_CLIENTE.isnull().sum()
base_2.COD_CLIENTE.isnull().sum()
base_3.COD_CLIENTE.isnull().sum()
base_4.COD_CLIENTE.isnull().sum()
base_5.COD_CLIENTE.isnull().sum()


# "---------------------------------------------------------"
# "--------------------UNIENDO DATAFRAMES ------------------"
# "---------------------------------------------------------"

def joinColumns(df1, df2):
    df = df1.merge(df2, on = "COD_CLIENTE", how = "left")
    df = df.loc[(df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"]) |
                (df["MES_COTIZACION_y"].isnull()), :]
    df = df.sort_values("MES_COTIZACION_y", ascending = False)
    df = df.drop_duplicates(["COD_CLIENTE"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})
    return df

train = joinColumns(base_1, base_2)

def joinColumns2(df1, df2):

    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    dfx = df.loc[df["MES_DATA"] <= df["MES_COTIZACION_x"], :]
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_DATA"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]
    # print(dfx.dtypes)
    # print(dfy.dtypes)
    # print(dfz.dtypes)
    df = pd.concat([dfx, dfy, dfz], sort = False).drop("_merge", axis = 1)
    df = df.sort_values("MES_DATA", ascending = False)    
    df = df.drop_duplicates(["COD_CLIENTE"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})

    return df

train = joinColumns2(train, base_3)

def joinColumns3(df1, df2, agg_type = "num"):
    
    train_columns = df1.columns.to_list()
    
    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    dfx = df.loc[df["MES_DATA_y"] <= df["MES_COTIZACION_x"], :]
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_DATA_y"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]
    df = pd.concat([dfx, dfy, dfz], sort = False).drop("_merge", axis = 1)
    
    df.columns = train_columns + df.columns[len(train_columns):].to_list()

    # Separación entre predictores numéricos y categóricos

    nume = df.columns[(df.dtypes == "int64") | (df.dtypes == "float64")].to_list()
    nume = [col for col in nume if col not in train_columns]
    # print("Columnas numéricas")
    # print(nume)
    
    cate = df.columns[(df.dtypes == "category") | (df.dtypes == "object")].to_list()
    cate = [col for col in cate if col not in train_columns]
    # print("Columnas categóricas")
    # print(cate)
    
    df_num = df[["COD_CLIENTE"] + nume]
    df_cate = df[["COD_CLIENTE"] + cate]
    try:
        df_gr_m_num = df_num.groupby("COD_CLIENTE", as_index = False).mean()
        if agg_type == "num":
            df1 = df1.merge(df_gr_m_num, on = "COD_CLIENTE", how = "left")
    except Exception as e:
        print(e)
        pass
    
    try: 
        df_gr_c_cate = df_cate.groupby("COD_CLIENTE", as_index = False).count()
        if agg_type == "cat":
            df1 = df1.merge(df_gr_c_cate, on = "COD_CLIENTE", how = "left")
        
    except Exception as e:
        print(e)
        pass
    
    return df1

train = joinColumns3(train, base_4, "num")
train = joinColumns3(train, base_5, "num")

train.MES_COTIZACION = train.MES_COTIZACION + pd.offsets.MonthBegin(0)

# "---------------------------------------------------------"

# Dividiendo columnas por tipo

label_cols = ["FLG_DESEMBOLSO"]
label = 'FLG_DESEMBOLSO'

numeric_cols = train.dtypes[train.dtypes == np.float64].index.to_list() +\
    train.dtypes[train.dtypes == np.int64].index.to_list()
numeric_cols = [col for col in numeric_cols if col not in label_cols]

cat_cols = train.dtypes[(train.dtypes == "O") | (train.dtypes == "category")].index[2:].to_list()

dt_range = pd.date_range(train.MES_COTIZACION.min(), train.MES_COTIZACION.max(), freq = "1MS")



for month in dt_range:
    print(month)
    stages = []
    train_temp = train[train.MES_COTIZACION <= month]

    # Feature Selection 
    
    # Dropping Null Columns
    
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

    #-----------------------------------------
    # codigo Uri

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
    target = train_temp[label]
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
        t_ctr += 1
        
    t_selec = pd.DataFrame(t_sel, index = train_num.columns)
    
    #-----------------------------------------
    
    # codigo Quique
    
    # ## Análisis Columnas categóricas
    
    # for col in cat_cols:
    #     train[col].value_counts().plot.bar(title = col)
    #     s = "\nMissing Values: " + str(train.isnull().sum()[col]) + \
    #         "\nMissing Values %: " + str(round(train.isnull().sum()[col]/len(train),4))
    #     plt.figtext(1, 0.5, s)
    #     plt.show()
    
    # Feature Selection con Chi Cuadrada
    
    df = train[cat_cols + label_cols].copy()
    for col in cat_cols:
        df.loc[df[col].isnull(), col] = "NaN"
        df[col] = LabelEncoder().fit_transform(df[col])
    
    X = df.drop([label], axis = 1)
    y = df[label]
    chi_scores = chi2(X,y)
    p_values = pd.Series(chi_scores[1], index = X.columns)
    droped_chi2_cols = p_values[p_values > 0.05].to_list()

    p_values.sort_values(ascending = False , inplace = True)
    p_values.plot.bar()
    plt.show()
    
    droped_cols = droped_null_cols + droped_corr_cols + droped_ttest_cols + droped_chi2_cols
    # print(droped_cols)
    
    final_num_cols = [col for col in numeric_cols if col not in droped_cols]
    final_cat_cols = [col for col in cat_cols if col not in droped_cols]
    
    feature_selector = Pipeline(steps = [
        ("null_dropper", MultiColumnDropper(droped_cols))])
    
    # Imputer categóricas con más frecuente
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
        # ('scaler', StandardScaler())
        ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("feature_select", feature_selector, droped_cols),
            ('num', numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)])
        
    param_grid_lr = {
        'classifier__C': [0.1, 1.0, 10, 100],
    }

    param_grid_rf = {
        'classifier__n_estimators': [150, 200]
    }

    lr_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression())])    
    

    rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_jobs = -1))])    
    
    grid_search = GridSearchCV(rf_clf, param_grid_rf, cv = 5, n_jobs = -1, scoring = "accuracy")
    
    x_train = train_temp[[col for col in train_temp.columns if col != label]]
    y_train = train_temp[label]

    grid_search.fit(x_train, y_train)


    #-----------------------------------------
    
    # evaluator 
    evaluation_df = pd.DataFrame(grid_search.cv_results_)
    print(grid_search.best_score_)
    # roc_curve plot

