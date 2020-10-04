package edu.missouri.Constants

object Constants {
  // Generic Constants.
  val APP_NAME = "XGBoost-Spark"
  val SEED = 123
  val CLASSIFICATION_DATASET = "data/IrisDataset.csv"
  val REGRESSION_DATASET = "data/ServoDataset.csv"
  val CLASSIFICATION_MODEL_DIR = "model/Iris"
  val REGRESSION_MODEL_DIR = "model/Servo"

  // Iris Dataset Column Names.
  val CLASSIFICATION_COL_1 = "sepal length"
  val CLASSIFICATION_COL_2 = "sepal width"
  val CLASSIFICATION_COL_3 = "petal length"
  val CLASSIFICATION_COL_4 = "petal width"
  val CLASSIFICATION_COL_5 = "class"
  val FEATURE_OUTPUT_COL = "fetures"
  val LABEL_OUTPUT_COL = "classIndex"

  // Servo Dataset Column Names.
  val REGRESSION_COL_1 = "motor"
  val REGRESSION_COL_2 = "screw"
  val REGRESSION_COL_3 = "pgain"
  val REGRESSION_COL_4 = "vgain"
  val REGRESSION_COL_5 = "class"
  val OHE_OUTPUT_COL_1 = "motor_one"
  val OHE_OUTPUT_COL_2 = "screw_one"

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
