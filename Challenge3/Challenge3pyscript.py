import os

from pyspark import SparkContext
from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim import *
from bigdl.optim.optimizer import *
from bigdl.dlframes.dl_classifier import *
import bigdl.version

# create sparkcontext with bigdl configuration
init_engine() # prepare the bigdl environment 
bigdl.version.__version__ # Get the current BigDL version

import pandas as pd
import numpy as np

from pyspark.sql import SQLContext, SparkSession, Row, DataFrame, Window
from pyspark.sql.functions import when, count, col, isnan, isnull, mean, lit, regexp_replace, abs as abs_spark, sum as sum_spark, rand, monotonically_increasing_id, row_number
from pyspark.sql.types import StringType, DoubleType, BooleanType, DateType, IntegerType, StructType, StructField, LongType,ByteType, ArrayType
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer, ChiSqSelector, OneHotEncoder, VectorAssembler, Imputer, StandardScaler, StringIndexer, OneHotEncoderEstimator, NGram, HashingTF, MinHashLSH, RegexTokenizer
from pyspark.ml.classification import LogisticRegression, LinearSVC, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from functools import reduce
import itertools

# dir(SparkSession.builder)
spark = SparkSession.builder.appName("ClasificacionBasemodelo").getOrCreate()
spark.sparkContext.getConf().getAll()

import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
from sklearn import datasets
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier

# Nuevo SCHEMA para no modificar el anterio por si nos funciona
schema = StructType([
    StructField("CD_CLIENTE", StringType()),
    StructField("CD_DICTAMEN", DoubleType()),
    StructField("CD_TIPOLOGIA", StringType()), # tipología PLD
    StructField("SEG_MOD", StringType()),
    StructField("DES_RIE_TIP", DoubleType()), # falta en diccionario
    StructField("DES_COT_BMV", StringType()), # falta en diccionario
    StructField("DES_ID_SEC", StringType()),
    StructField("DES_ID_SEG", StringType()),
    StructField("DES_ID_EDO", StringType()),
    StructField("DES_ID_BAN", StringType()),
    StructField("DES_ID_RB", StringType()),
    StructField("DES_EDA", DoubleType()),
    StructField("DES_ANT_CTE", DoubleType()),
    StructField("DES_ANT_CTA", DoubleType()),
    StructField("DES_MON_DEC", DoubleType()), # ¿por qué son números enteros?
    StructField("EFE_NU_MES_MON0", DoubleType()),
    StructField("EFE_NU_MES_MON1", DoubleType()),
    StructField("EFE_NU_MES_MOND", DoubleType()), # ¿Por qué no son números enteros? ¿Qué es abono-cargo?
    StructField("EFE_SD_MON0", DoubleType()),
    StructField("EFE_SD_MON1", DoubleType()),
    StructField("EFE_SD_MOND", DoubleType()),
    StructField("EFE_NU_MES_MOV0", DoubleType()),
    StructField("EFE_NU_MES_MOV1", DoubleType()),
    StructField("EFE_SD_MOV0", DoubleType()),
    StructField("EFE_SD_MOV1", DoubleType()),
    StructField("OPI_NU_MES_MON0", DoubleType()),
    StructField("OPI_NU_MES_MON1", DoubleType()),
    StructField("OPI_NU_MES_MOND", DoubleType()),
    StructField("OPI_SD_MON0", DoubleType()),
    StructField("OPI_SD_MON1", DoubleType()),
    StructField("OPI_SD_MOND", DoubleType()),
    StructField("OPI_NU_MES_MOV0", DoubleType()),
    StructField("OPI_NU_MES_MOV1", DoubleType()),
    StructField("OPI_SD_MOV0", DoubleType()),
    StructField("OPI_SD_MOV1", DoubleType()),
    StructField("OPI_NU_PAI0", DoubleType()),
    StructField("OPI_NU_PAI1", DoubleType()),
    StructField("OPI_SUM_RIE0", DoubleType()),
    StructField("OPI_SUM_RIE1", DoubleType()),
    StructField("OPI_NU_PAI_PAR0", DoubleType()), # PAR: paraíso fiscal
    StructField("OPI_NU_PAI_PAR1", DoubleType()), # ¿Por qué no son números enteros?
    StructField("OPI_PON_MON_PAR0", DoubleType()),
    StructField("OPI_PON_MON_PAR1", DoubleType()),
    StructField("MEC_NU_MES_MON0", DoubleType()),
    StructField("MEC_NU_MES_MON1", DoubleType()),
    StructField("MEC_SD_MON0", DoubleType()),
    StructField("MEC_SD_MON1", DoubleType()),
    StructField("MEC_SUM_RIE0", DoubleType()),
    StructField("MEC_SUM_RIE1", DoubleType()),
    StructField("MEC_PON_MON0", DoubleType()),
    StructField("MEC_PON_MON1", DoubleType()),
    StructField("TER_PON_MON0", DoubleType()), # TER: tercero
    StructField("TER_PON_MON1", DoubleType()), 
    StructField("BTS_NU_MES_MON1", DoubleType()),
    StructField("BTS_SD_MON1", DoubleType()), 
    StructField("CHP_NU_MES_MON0", DoubleType()),
    StructField("CHP_NU_MES_MON1", DoubleType()),
    StructField("CHP_SD_MON0", DoubleType()),
    StructField("CHP_SD_MON1", DoubleType()),
    StructField("CHC_NU_MES_MON0", DoubleType()),
    StructField("CHC_NU_MES_MON1", DoubleType()),
    StructField("CHC_SD_MON0", DoubleType()),
    StructField("CHC_SD_MON1", DoubleType()),
    StructField("SPE_NU_MES_MON0", DoubleType()),
    StructField("SPE_NU_MES_MON1", DoubleType()),
    StructField("SPE_SD_MON0", DoubleType()),
    StructField("SPE_SD_MON1", DoubleType()),
    StructField("SPE_PON_MON0", DoubleType()),
    StructField("SPE_PON_MON1", DoubleType()),
    StructField("SPI_NU_MES_MON0", DoubleType()),
    StructField("SPI_NU_MES_MON1", DoubleType()),
    StructField("SPI_SD_MON0", DoubleType()),
    StructField("SPI_SD_MON1", DoubleType()),
    StructField("DES_ID_PAI_NAC", StringType()),
    StructField("DES_ID_OCU", StringType()),
    StructField("DES_ID_SOC", StringType()),
    StructField("DES_GPO", StringType()), # Grupo empresarial
    StructField("DES_ACT_VUL", StringType()),
    StructField("DES_RSK_SEG", DoubleType()),
    StructField("DES_RSK_GES", DoubleType()),
    StructField("DES_RSK_ALT", DoubleType()),
    StructField("DES_RSK_NAC", DoubleType()), # Podría se numérico o categórico: 0, 1, 3, 5, 7, 9, 11
    StructField("DES_DIS_ALT_GES", DoubleType()),
    StructField("DES_RSK_SEC", DoubleType()),
    StructField("DES_RSK_ACT", DoubleType()),
    StructField("DES_RSK_OCU", DoubleType()),
    StructField("DES_RSK_SOC", DoubleType()),
    StructField("DES_NU_INT", DoubleType()),
    StructField("DES_NU_CAP", DoubleType()),
    StructField("DES_NU_EMP", DoubleType()),
    StructField("DES_NU_PRO", DoubleType()),
    StructField("DES_NU_SEG", DoubleType()),
    StructField("EFE_NU_MOV", DoubleType()),
    StructField("EFE_NU_MES", DoubleType()),
    StructField("EFE_AVG_MON", DoubleType()),
    StructField("EFE_AVG_MON0", DoubleType()),
    StructField("EFE_AVG_MON1", DoubleType()),
    StructField("EFE_AVG_MONV", DoubleType()),
    StructField("EFE_AVG_MONC", DoubleType()),
    StructField("EFE_AVG_MONM", DoubleType()),
    StructField("EFE_AVG_MONR", DoubleType()),
    StructField("EFE_MAX_OFI", DoubleType()),
    StructField("EFE_MAX_DIS", DoubleType()),
    StructField("EFE_MON_USD_FN", DoubleType()), # FN: frontera norte
    StructField("EFE_MON_MXP_FN", DoubleType()),
    StructField("EFE_MON_USD_AR", DoubleType()), # AR: alto riesgo
    StructField("EFE_MON_MXP_AR", DoubleType()), 
    StructField("OPI_NU_MOV0", DoubleType()),
    StructField("OPI_NU_MOV1", DoubleType()),
    StructField("OPI_NU_MES0", DoubleType()),
    StructField("OPI_NU_MES1", DoubleType()),
    StructField("OPI_AVG_MON0", DoubleType()),
    StructField("OPI_AVG_MON1", DoubleType()),
    StructField("OPI_NU_CTA0", DoubleType()),
    StructField("OPI_NU_CTA1", DoubleType()),
    StructField("OPI_MAX_PAI0", DoubleType()),
    StructField("OPI_MAX_PAI1", DoubleType()),
    StructField("OPI_NU_PAI00", DoubleType()), # no está en diccionario
    StructField("OPI_NU_PAI11", DoubleType()), # no está en diccionario
    StructField("TER_NU_MES0", DoubleType()),
    StructField("TER_NU_MES1", DoubleType()),
    StructField("TER_AVG_MON0", DoubleType()),
    StructField("TER_AVG_MON1", DoubleType()),
    StructField("TER_NU_MOV_LI0", DoubleType()),
    StructField("TER_NU_MOV_LI1", DoubleType()),
    StructField("TER_SUM_MON_LI0", DoubleType()),
    StructField("TER_SUM_MON_LI1", DoubleType()),
    StructField("TER_NU_CTE_LI0", DoubleType()),
    StructField("TER_NU_CTE_LI1", DoubleType()),
    StructField("BTS_NU_MES1", DoubleType()),
    StructField("BTS_AVG_MON1", DoubleType()),
    StructField("CHP_NU_MES0", DoubleType()),
    StructField("CHP_NU_MES1", DoubleType()),
    StructField("CHP_AVG_MON_EFE", DoubleType()),
    StructField("CHP_AVG_MON0", DoubleType()),
    StructField("CHP_NU_MOV_LI0", DoubleType()),
    StructField("CHP_SUM_MON_LI0", DoubleType()),
    StructField("CHP_NU_CTE_LI0", DoubleType()),
    StructField("CHC_NU_MES0", DoubleType()),
    StructField("CHC_NU_MES1", DoubleType()),
    StructField("CHC_AVG_MON0", DoubleType()),
    StructField("CHC_AVG_MON1", DoubleType()),
    StructField("CHC_AVG_MON_OTR", DoubleType()),
    StructField("CHC_SUM_MON_LI0", DoubleType()),
    StructField("CHC_SUM_MON_LI1", DoubleType()),
    StructField("CHC_NU_CTE_LI0", DoubleType()),
    StructField("CHC_NU_CTE_LI1", DoubleType()),
    StructField("SPE_NU_MES0", DoubleType()),
    StructField("SPE_NU_MES1", DoubleType()),
    StructField("SPE_AVG_MON0", DoubleType()),
    StructField("SPE_AVG_MON1", DoubleType()),
    StructField("SPE_NU_CTA0", DoubleType()),
    StructField("SPE_NU_CTA1", DoubleType()),
    StructField("SPE_SUM_MON_LI0", DoubleType()),
    StructField("SPE_SUM_MON_LI1", DoubleType()),
    StructField("SPE_NU_CTE_LI0", DoubleType()),
    StructField("SPE_NU_CTE_LI1", DoubleType()),
    StructField("SPI_NU_MES0", DoubleType()),
    StructField("SPI_NU_MES1", DoubleType()),
    StructField("SPI_AVG_MON0", DoubleType()),
    StructField("SPI_AVG_MON1", DoubleType()),
    StructField("SPI_NU_CTA0", DoubleType()),
    StructField("SPI_NU_CTA1", DoubleType()),
    StructField("CCC_NU_CTE_LI", DoubleType()),
])


df = df.drop("OPI_NU_PAI_PAR1")
df = df.drop("EFE_NU_MES_MOND")