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
