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
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer

test = pd.read_csv(r"./data/Base1_test.csv")

dateparse = lambda x: pd.to_datetime(x)

# Carga datos
base_1 = pd.read_csv(r"./data/Base1_train.csv", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Base cotizaciones
base_2 = pd.read_csv(r"./data/Base2.csv", sep = ";", parse_dates = ["MES_COTIZACION"], date_parser = dateparse) # Información sociodemográfica + digital
base_3 = pd.read_csv(r"./data/Base3.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base productos BBVA
base_4 = pd.read_csv(r"./data/Base4.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de saldos en el Sistema Financiero
base_5 = pd.read_csv(r"./data/Base5.csv", sep = ";", parse_dates = ["MES_COTIZACION", "MES_DATA"], date_parser = dateparse) # Base de consumos con tarjeta

base_4["ST_CREDITO"] = base_4["ST_CREDITO"].astype(str)
base_5[["CD_DIVISA", "TP_TARJETA"]] = base_5[["CD_DIVISA", "TP_TARJETA"]].astype(str)

# Exploratory Data Analysis
## Base_1

#Cantidad de clientes unicos / longitud del df
len(base_1.COD_CLIENTE.unique()) / base_1.shape[0]
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
    df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})
    return df

train = joinColumns(base_1, base_2)

def joinColumns2(df1, df2):

    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    dfx = df.loc[df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"], :]
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]
    # print(dfx.dtypes)
    # print(dfy.dtypes)
    # print(dfz.dtypes)
    df = pd.concat([dfx, dfy, dfz], sort = False).drop("_merge", axis = 1)
    df = df.sort_values("MES_DATA", ascending = False)    
    df = df.drop_duplicates(["COD_CLIENTE", "COD_SOL"], keep = "first")
    df = df.drop("MES_COTIZACION_y", axis = 1)
    df = df.rename(columns = {"MES_COTIZACION_x": "MES_COTIZACION"})

    return df

train = joinColumns2(train, base_3)

def joinColumns3(df1, df2, agg_type = "num"):
    
    train_columns = df1.columns.to_list()
    
    df = df1.merge(df2, on = "COD_CLIENTE", how = "left", indicator = True)
    dfx = df.loc[df["MES_COTIZACION_y"] <= df["MES_COTIZACION_x"], :]
    dfy = df.loc[df['_merge'] == 'left_only', :]
    dfz = df.loc[df["MES_COTIZACION_y"] > df["MES_COTIZACION_x"], df.columns[:len(df1.columns)]]
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
    
    df_num = df[["COD_SOL"] + nume]
    df_cate = df[["COD_SOL"] + cate]
    try:
        df_gr_m_num = df_num.groupby("COD_SOL", as_index = False).mean()
        if agg_type == "num":
            df1 = df1.merge(df_gr_m_num, on = "COD_SOL", how = "left")
    except Exception as e:
        print(e)
        pass
    
    try: 
        df_gr_c_cate = df_cate.groupby("COD_SOL", as_index = False).count()
        if agg_type == "cat":
            df1 = df1.merge(df_gr_c_cate, on = "COD_SOL", how = "left")
        
    except Exception as e:
        print(e)
        pass
    
    return df1

train = joinColumns3(train, base_4, "num")
train = joinColumns3(train, base_5, "num")

# "---------------------------------------------------------"

# Dividiendo columnas por tipo
bool_cols = [
"FLG_DESEMBOLSO",
"USO_BI_M0",
"USO_BI_M1",
"USO_BI_M2",
"USO_BM_M0",
"USO_BM_M1",
"USO_BM_M2"
]

# for col in bool_cols:
#     train[col] = train[col].astype(bool)

numeric_cols = train.dtypes[train.dtypes == np.float64].index.to_list() +\
    train.dtypes[train.dtypes == np.int64].index.to_list()

cat_cols = train.dtypes[(train.dtypes == "O") | (train.dtypes == "category")].index[2:].to_list()

# Análisis Columnas numéricas

# Eliminando una de cada dos columnas numéricas correlacionadas
# (Para facilitar análisis)

corrmat = train[numeric_cols].corr()
droped_corr_cols = []

for col in corrmat.columns:
    if col not in droped_corr_cols:
        a = corrmat[col][(abs(corrmat[col]) > 0.9) & (abs(corrmat[col]) < 1)]
        for colname in a.index:
            if colname in droped_corr_cols:
                pass
            else:
                print(colname + " se correlaciona mucho con " + col +
                      "con un PCC de " + str(corrmat.loc[colname, col]) + 
                      "\n por lo tanto nos deshacemos de ella por aportar\n la misma información")
                droped_corr_cols.append(colname)
                corrmat = corrmat.drop(colname, axis = 1)

# Graficando histogramas de columnas numéricas

for col in numeric_cols:
    train[col].plot.hist(title = col)
    s = train.describe()[col].to_string() + \
        "\nMissing Values: " + str(train.isnull().sum()[col]) + \
        "\nMissing Values %: " + str(round(train.isnull().sum()[col]/len(train),4))
    plt.figtext(1, 0.5, s)
    plt.show()

## Análisis Columnas categóricas

for col in cat_cols:
    train[col].value_counts().plot.bar(title = col, rot = 0)
    # s = train.describe()[col].to_string() + \
    #     "\nMissing Values: " + str(train.isnull().sum()[col]) + \
    #     "\nMissing Values %: " + str(round(train.isnull().sum()[col]/len(train),4))
    # plt.figtext(1, 0.5, s)
    plt.show()

imp = SimpleImputer(strategy="most_frequent")
imp.fit_transform(df)

# Chi Cuadrada

for col in cat_cols:
    label_encoder = LabelEncoder()
    train[col] = label_encoder.fit_transform(train[col])

label = 'FLG_DESEMBOLSO'

X = train[cat_cols + bool_cols].drop(label,axis=1)
y = train[label]

chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar()