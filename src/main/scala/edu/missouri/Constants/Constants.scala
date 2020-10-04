package edu.missouri.Constants

object Constants {
  // Generic Constants.
  val APP_NAME = "XGBoost-Spark"
  val SEED = 123
  val CLASSIFICATION_DATASET = "data/IrisDataset.csv"
  val REGRESSION_DATASET = "data/AutomobileDataset.csv"
  val CLASSIFICATION_MODEL_DIR = "model/Iris"
  val REGRESSION_MODEL_DIR = "model/Automobile"

  // Colum Names.
  val COL_1 = "sepal length"
  val COL_2 = "sepal width"
  val COL_3 = "petal length"
  val COL_4 = "petal width"
  val COL_5 = "class"
  val FEATURE_OUTPUT_COL = "fetures"
  val LABEL_OUTPUT_COL = "classIndex"

  // XGBoost Hyperparameters.
  val ETA = 0.1f
  val MAX_DEPTH = 2
  val OBJECTIVE = "multi:softprob"
  val NUM_CLASS = 3
  val NUM_ROUND = 100
  val TREE_METHOD = "auto"
  val NUM_WORKERS = 2
  val CV_FOLDS = 3

  // Label Converter Constants
  val PREDICTION = "prediction"
  val LABEL = "realLabel"
}
