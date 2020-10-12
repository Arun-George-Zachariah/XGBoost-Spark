# XGBoost-Spark
XGBoost stands for e**X**treme **G**radient **Boost**ing. It is a highly optimized and distributed implementation of gradient boosting decision trees, to handle large and complicated datasets. In this project, we would we using XGBoost4J-Spark to build simple Classification and Regression models.

## Setup
Requirements:
* [JDK](https://www.java.com/en/download/)
* [Scala](https://www.scala-lang.org/download/)
* [sbt](https://www.scala-sbt.org/download.html)
* [Apache Spark](https://spark.apache.org/downloads.html)

On Ubuntu, you can use `scripts/setup.sh` to setup the pre-requisites.

## Build
```
sbt clean assembly
```

## Execute
* Classification
```
spark-submit --class edu.missouri.XGBoost.ClassifierPipeline target/scala-2.12/XGBoost-Spark-assembly-0.1.jar
```
* Regression
```
spark-submit --class edu.missouri.XGBoost.RegressionPipeline target/scala-2.12/XGBoost-Spark-assembly-0.1.jar
```

## References:
* [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html)
* [DMLC XGBoost ](https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j-example/src/main/scala/ml/dmlc/xgboost4j/scala/example/spark/SparkMLlibPipeline.scala)
* [XGBoost Regression with Spark DataFrames](https://docs.databricks.com/_static/notebooks/xgboost-regression.html)
* [Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/Automobile)
* [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/Iris)