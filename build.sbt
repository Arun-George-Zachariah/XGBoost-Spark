name := "XGBoost-Spark"

version := "0.1"

scalaVersion := "2.12.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "2.4.0",
  "org.apache.spark" %% "spark-mllib" % "3.0.1",
  "ml.dmlc" %% "xgboost4j-spark" % "1.2.0"
)

assemblyMergeStrategy in assembly := {
  case PathList ("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}