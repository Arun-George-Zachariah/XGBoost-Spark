package edu.missouri.XGBoost

import edu.missouri.Constants.Constants
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object RegressionPipeline {
  def main(args: Array[String]): Unit = {

    // Creating the spark session, which is the entry point for any Spark program dealing with a data frame.
    val spark = SparkSession.builder().appName(Constants.APP_NAME).getOrCreate();

    // Defining the data schema.
    val schema = new StructType(Array(
      StructField(Constants.REGRESSION_COL_1, StringType, true),
      StructField(Constants.REGRESSION_COL_2, StringType, true),
      StructField(Constants.REGRESSION_COL_3, DoubleType, true),
      StructField(Constants.REGRESSION_COL_4, DoubleType, true),
      StructField(Constants.REGRESSION_COL_5, DoubleType, true)
    ))

    // Loading the data.
    val inputData = spark.read.schema(schema).csv(Constants.REGRESSION_DATASET)

    // Splitting the dataset into test and train directories.
    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2), Constants.SEED)

    // Building the ML Pipeline - Start
    // 1) Converting string labels to indexed double label
    val stringIndexer = new StringIndexer()
      .setInputCols(Array(Constants.REGRESSION_COL_1, Constants.REGRESSION_COL_2))
      .setOutputCols(Array(Constants.REGRESSION_COL_1 + Constants.INDEX_COL, Constants.REGRESSION_COL_2 + Constants.INDEX_COL))

    // 2) Performing One Hot Encoding.
    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(Array(Constants.REGRESSION_COL_1 + Constants.INDEX_COL, Constants.REGRESSION_COL_2 + Constants.INDEX_COL))
      .setOutputCols(Array(Constants.REGRESSION_COL_1 + Constants.OHE_COL, Constants.REGRESSION_COL_2 + Constants.OHE_COL))
      .setDropLast(false)

    // 3) Assembling all features into a single vector column.
    val assembler = new VectorAssembler()
      .setInputCols(Array(Constants.REGRESSION_COL_1 + Constants.INDEX_COL, Constants.REGRESSION_COL_2 + Constants.INDEX_COL, Constants.REGRESSION_COL_3, Constants.REGRESSION_COL_4))
      .setOutputCol(Constants.FEATURE_OUTPUT_COL)

    // 4) Using XGBoostRegressor to train the regression model.
    val booster = new XGBoostRegressor(
      Map("eta" -> Constants.ETA,
        "max_depth" -> Constants.MAX_DEPTH,
        "objective" -> Constants.REGRESSION_OBJECTIVE,
        "num_round" -> Constants.NUM_ROUND,
        "num_workers" -> Constants.NUM_WORKERS,
        "tree_method" -> Constants.TREE_METHOD
      )
    )
    booster.setFeaturesCol(Constants.FEATURE_OUTPUT_COL)
    booster.setLabelCol(Constants.REGRESSION_COL_5)

    // Pipeline with the sequence of stages
    val pipeline = new Pipeline().setStages(Array(stringIndexer, oneHotEncoder, assembler, booster))

    // Building the ML Pipeline - End

    // Fitting the pipeline to the input dataset i.e training the model.
    val model = pipeline.fit(train)

    // Performing prediction.
    val predict = model.transform(test)

    // Evaluating the model and computing the accuracy.
    val evaluator = new RegressionEvaluator().setLabelCol(Constants.REGRESSION_COL_5).setPredictionCol(Constants.PREDICTION)
    val accuracy = evaluator.evaluate(predict)
    println("The model accuracy is : " + accuracy)

    // Tuning the model using CrossValidation.
    val paramGrid = new ParamGridBuilder()
      .addGrid(booster.maxDepth, Array(3, 8))
      .addGrid(booster.eta, Array(0.2, 0.6))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(Constants.CV_FOLDS)

    // Training the model.
    val cvModel = cv.fit(train)

    // Obtaining the best model.
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[XGBoostRegressionModel]
    println("The training summary of best XGBoostRegressionModel : " + bestModel.summary)

    // Saving the best model.
    model.write.overwrite().save(Constants.REGRESSION_MODEL_DIR)

    // Load a saved model and serving
    val model2 = PipelineModel.load(Constants.REGRESSION_MODEL_DIR)
    model2.transform(test).show(false)
  }
}