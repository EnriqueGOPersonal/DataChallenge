# #!/bin/sh
# export BIGDL_HOME="/home/MB58091/.local/lib/python3.5/site-packages/bigdl/share"
# BIGDL_JAR_NAME=$(ls $BIGDL_HOME/lib/ | grep jar-with-dependencies.jar)
# export BIGDL_JAR_NAME
# export BIGDL_JAR=$BIGDL_HOME/lib/$BIGDL_JAR_NAME
# BIGDL_PY_ZIP_NAME=$(ls $BIGDL_HOME/lib/ | grep python-api.zip)
# export BIGDL_PY_ZIP_NAME
# export BIGDL_PY_ZIP=$BIGDL_HOME/lib/$BIGDL_PY_ZIP_NAME
# export BIGDL_CONF=$BIGDL_HOME/conf/spark-bigdl.conf
# PYTHONPATH=$BIGDL_PY_ZIP:$PYTHONPATH
# export PYTHONPATH

import os

os.environ["BIGDL_HOME"]="/home/MB58091/.local/lib/python3.5/site-packages/bigdl/share"
BIGDL_JAR_NAME = !echo $(ls $BIGDL_HOME/lib/ | grep jar-with-dependencies.jar)
os.environ["BIGDL_JAR_NAME"]= BIGDL_JAR_NAME[0]
BIGDL_JAR = !echo $BIGDL_HOME/lib/$BIGDL_JAR_NAME
os.environ["BIGDL_JAR"] = BIGDL_JAR[0]
BIGDL_PY_ZIP_NAME = !echo $(ls $BIGDL_HOME/lib/ | grep python-api.zip)
os.environ["BIGDL_PY_ZIP_NAME"] = BIGDL_PY_ZIP_NAME[0]
BIGDL_PY_ZIP = !echo $BIGDL_HOME/lib/$BIGDL_PY_ZIP_NAME
os.environ["BIGDL_PY_ZIP"]= BIGDL_PY_ZIP[0]
BIGDL_CONF = !echo $BIGDL_HOME/conf/spark-bigdl.conf
os.environ["BIGDL_CONF"] = BIGDL_CONF[0]
PYTHONPATH = !echo $BIGDL_PY_ZIP:$PYTHONPATH
os.environ["PYTHONPATH"]= PYTHONPATH[0] 

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession

os.environ["SPARK_CONF_DIR"] = os.environ["BIGDL_CONF"]

spark = SparkSession.builder.addPyFile(os.environ["BIGDL_PY_ZIP"]).config("spark.jars`, os.environ["BIGDL_JAR"]).\
config("spark.driver.extraClassPath", os.environ["BIGDL_JAR"]).config("spark.EXECUTOR.extraClassPath", os.environ["BIGDL_JAR"]).\
getOrCreate()
