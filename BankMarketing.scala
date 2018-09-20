// Databricks notebook source
display(dbutils.fs.ls("/FileStore/tables/"))

// COMMAND ----------

val df = spark.read
  .option("header","true")
  .option("inferSchema", "true")
  .format("csv")
  .load("dbfs:/FileStore/tables/mb_full-d0762.csv")
display(df)


// COMMAND ----------

df.printSchema()

// COMMAND ----------

/*import org.apache.spark.sql.functions.{count, sum, when}
val df1 = df.withColumn("beg_month", when($"day" <= "10", 1).otherwise(0))
val df2 = df1.withColumn("mid_month", when($"day" <= "20" && $"day" > "10", 1).otherwise(0))
val df3 = df2.withColumn("end_month", when($"day" > "20", 1).otherwise(0))*/

// COMMAND ----------

/*val df4 = df3.withColumn("20_30_age", when($"age" < "30", 1).otherwise(0))
val df5 = df4.withColumn("30_40_age", when($"age" < "40" && $"age" >= "30", 1).otherwise(0))
val df6 = df5.withColumn("40_50_age", when($"age" < "50" && $"age" >= "40", 1).otherwise(0))
val df7 = df6.withColumn("50_60_age", when($"age" < "60" && $"age" >= "50", 1).otherwise(0))
val df8 = df7.withColumn("60_older_age", when($"age" >= "60", 1).otherwise(0))*/

// COMMAND ----------

// MAGIC %md ### EXPLORATORY ANALYSIS

// COMMAND ----------

df.count()

// COMMAND ----------

val resultDF = df.groupBy("y")
  //.pivot("y")
  .count()
display(resultDF)

// COMMAND ----------

val contactDF = df.groupBy("contact")
  .pivot("y")
  .count()
display(contactDF)

// COMMAND ----------

display(contactDF)

// COMMAND ----------



// COMMAND ----------

display(df.groupBy("poutcome").count())

// COMMAND ----------

// MAGIC %md Unknown is 36,959 -  82%

// COMMAND ----------

val outDF = df.groupBy("poutcome")
  .pivot("y")
  .count()

outDF.show()

// COMMAND ----------

display(outDF)

// COMMAND ----------

// MAGIC %md If the previous outcime was succes, there is 50% chanse to get possitive result.

// COMMAND ----------

val dayDF = df.groupBy("day")
  .pivot("y")
  .count()

display(dayDF)

// COMMAND ----------

display(dayDF)

// COMMAND ----------

val durDF = df.groupBy("duration")
  .pivot("y")
  .count()

display(durDF)

// COMMAND ----------

display(durDF)

// COMMAND ----------

display(durDF)

// COMMAND ----------

val monthDF = df.groupBy("month")
  .pivot("y")
  .count()

display(monthDF)

// COMMAND ----------

display(monthDF)

// COMMAND ----------

display(monthDF)


// COMMAND ----------

val ageDF = df.groupBy("age")
  .pivot("y")
  .count()

// COMMAND ----------

display(ageDF)

// COMMAND ----------

val housDF = df.groupBy("housing")
  .pivot("y")
  .count()

// COMMAND ----------

display(housDF)

// COMMAND ----------

val eduDF = df.groupBy("education")
  .pivot("y")
  .count()

// COMMAND ----------

display(eduDF)

// COMMAND ----------

val defDF = df.groupBy("default")
  .pivot("y")
  .count()

// COMMAND ----------

display(defDF)

// COMMAND ----------

val jobDF = df.groupBy("job")
  .pivot("y")
  .count()

//jobDF.show()
display(jobDF)

// COMMAND ----------

val marDF = df.groupBy("marital")
  .pivot("y")
  .count()
display(marDF)

// COMMAND ----------

val loanDF = df.groupBy("loan")
  .pivot("y")
  .count()
display(loanDF)

// COMMAND ----------

// MAGIC %md ### Extracting, transforming and selecting features

// COMMAND ----------

import org.apache.spark.ml.feature.{OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.feature.StringIndexer

val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
val marIndexer = new StringIndexer().setInputCol("marital").setOutputCol("marIndex")
val eduIndexer = new StringIndexer().setInputCol("education").setOutputCol("eduIndex")
val defIndexer = new StringIndexer().setInputCol("default").setOutputCol("defIndex")
val housIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housIndex")
val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex")
val contIndexer = new StringIndexer().setInputCol("contact").setOutputCol("contIndex")
val monIndexer = new StringIndexer().setInputCol("month").setOutputCol("monIndex")
val poutIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutIndex")
val yIndexer = new StringIndexer().setInputCol("y").setOutputCol("yIndex")

val oneHot = new OneHotEncoderEstimator()
  .setInputCols(Array("jobIndex","marIndex","eduIndex","defIndex","housIndex","loanIndex","contIndex","monIndex","poutIndex"))
  .setOutputCols(Array("jobVect","marVect","eduVect","defVect","housVect","loanVect","contVect","monVect","poutVect"))

val vectorAssembler = new VectorAssembler()
.setInputCols(Array("age","balance","day","duration","campaign","pdays","previous","jobVect","marVect","eduVect","defVect","housVect","loanVect","contVect","monVect","poutVect"))
  .setOutputCol("features")

import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)
/*val vectorAssembler = new VectorAssembler()
.setInputCols(Array("age","balance","beg_month","mid_month","end_month","duration","campaign","pdays","previous","jobVect","marVect","eduVect","defVect","housVect","loanVect","contVect","monVect","poutVect"))
  .setOutputCol("features")*/

// COMMAND ----------

// MAGIC %md ### Split Dataset

// COMMAND ----------

val Array(train, test) = df.randomSplit(Array(0.7, 0.3))
train.cache()
test.cache()

println(train.count())
println(test.count())

// COMMAND ----------

// MAGIC %md ### Logistic Regression

// COMMAND ----------

import org.apache.spark.ml.classification.LogisticRegression
val estimator = new LogisticRegression()
  .setElasticNetParam(0.5)
  .setLabelCol("yIndex")
  //.setFeaturesCol("features")
  .setFeaturesCol("scaledFeatures")

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col

val pipeline = new Pipeline()
  .setStages(Array(jobIndexer,marIndexer,eduIndexer,defIndexer,housIndexer,loanIndexer,contIndexer,monIndexer, poutIndexer, yIndexer, oneHot, vectorAssembler, scaler,estimator))//estimator))

val pipelineModel = pipeline.fit(train)

// COMMAND ----------

val holdout_test = pipelineModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "yIndex", 
    """CASE double(round(prediction)) = yIndex
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout_test)



// COMMAND ----------

val accur= holdout_test.selectExpr("sum(equal)/sum(1)")
display(accur)

// COMMAND ----------

import org.apache.spark.sql.functions.{count, sum, when}

val rate = holdout_test
  .groupBy('yIndex)
  .agg(
    (sum(when('prediction === 'yIndex, 1)) / count('yIndex)).alias("true prediction rate"),
    count('yIndex).alias("count")
  )
display(rate)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val rm = new MulticlassMetrics(
  holdout_test.select("prediction", "yIndex").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("Confusion Matrix:\n" + rm.confusionMatrix)
println("Accuracy: " + rm.accuracy)

val labels = rm.labels
labels.foreach { l =>
  println(s"Precision($l) = " + rm.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + rm.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + rm.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + rm.fMeasure(l))
}

// Weighted stats
println("Weighted precision: " + rm.weightedPrecision)
println("Weighted recall: " + rm.weightedRecall)
println(s"Weighted F1 score: " + rm.weightedFMeasure)
println("Weighted false positive rate: " + rm.weightedFalsePositiveRate)


// COMMAND ----------

val holdout_train = pipelineModel
  .transform(train)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "yIndex", 
    """CASE double(round(prediction)) = yIndex
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout_train)

// COMMAND ----------

import org.apache.spark.sql.functions.{count, sum, when}

val rate1 = holdout_train
  .groupBy('yIndex)
  .agg(
    (sum(when('prediction === 'yIndex, 1)) / count('yIndex)).alias("true prediction rate"),
    count('yIndex).alias("count")
  )

display(rate1)

// COMMAND ----------

val rmTrain = new MulticlassMetrics(
  holdout_train.select("prediction", "yIndex").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("Confusion Matrix:\n" + rmTrain.confusionMatrix)
println("Accuracy: " + rmTrain.accuracy)

val labels = rmTrain.labels
labels.foreach { l =>
  println(s"Precision($l) = " + rmTrain.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + rmTrain.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + rmTrain.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + rmTrain.fMeasure(l))
}

// Weighted stats
println("Weighted precision: " + rmTrain.weightedPrecision)
println("Weighted recall: " + rmTrain.weightedRecall)
println(s"Weighted F1 score: " + rmTrain.weightedFMeasure)
println("Weighted false positive rate: " + rmTrain.weightedFalsePositiveRate)

// COMMAND ----------

import spark.implicits._
val metricsLR= Seq(
  ("Accuracy",rm.accuracy,rmTrain.accuracy),
  ("Precision(0.0)",rm.precision(0),rmTrain.precision(0)),
  ("Precision(1.0)",rm.precision(1),rmTrain.precision(1)),
  ("Recall(0.0)",rm.recall(0),rmTrain.recall(0)),
  ("Recall(1.0)",rm.recall(1),rmTrain.recall(1)),
  ("FPR(0.0)",rm.falsePositiveRate(0),rmTrain.falsePositiveRate(0)),
  ("FPR(1.0)",rm.falsePositiveRate(1),rmTrain.falsePositiveRate(1)),
  ("F1-Score(0.0)",rm.fMeasure(0),rmTrain.fMeasure(0)),
  ("F1-Score(1.0)",rm.fMeasure(1),rmTrain.fMeasure(1))
).toDF("metrics", "Test","Train")

display(metricsLR)

// COMMAND ----------

// MAGIC %md ### LinearSVC

// COMMAND ----------

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.StandardScaler

val lsvc = new LinearSVC()
  .setMaxIter(100)
  .setRegParam(0.05)
  .setLabelCol("yIndex")
  .setFeaturesCol("scaledFeatures")

val LSVMpipeline = new Pipeline()
  .setStages(Array(jobIndexer,marIndexer,eduIndexer,defIndexer,housIndexer,loanIndexer,contIndexer,monIndexer, poutIndexer, yIndexer, oneHot, vectorAssembler, scaler, lsvc))

val LSVMmodel = LSVMpipeline.fit(train)

// COMMAND ----------

val holdout3_test = LSVMmodel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "yIndex", 
    """CASE double(round(prediction)) = yIndex
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout3_test)


// COMMAND ----------

val accur3 = holdout3_test.selectExpr("sum(equal)/sum(1)")
display(accur3)

// COMMAND ----------

import org.apache.spark.sql.functions.{count, sum, when}

val rate3 = holdout3_test
  .groupBy('yIndex)
  .agg(
    (sum(when('prediction === 'yIndex, 1)) / count('yIndex)).alias("true prediction rate"),
    count('yIndex).alias("count")
  )

display(rate3)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics
val rm3 = new MulticlassMetrics(
  holdout3_test.select("prediction", "yIndex").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("Confusion Matrix:\n " + rm3.confusionMatrix)
println("Accuracy: " + rm3.accuracy)

val labels = rm3.labels
labels.foreach { l =>
  println(s"Precision($l) = " + rm3.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + rm3.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + rm3.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + rm3.fMeasure(l))
}

// Weighted stats
println("Weighted precision: " + rm3.weightedPrecision)
println("Weighted recall: " + rm3.weightedRecall)
println(s"Weighted F1 score: " + rm3.weightedFMeasure)
println("Weighted false positive rate: " + rm3.weightedFalsePositiveRate)

// COMMAND ----------

// MAGIC %md ### GBT Classifier

// COMMAND ----------

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}

// COMMAND ----------

val gbt = new GBTClassifier()
      .setLabelCol("yIndex")
      .setFeaturesCol("features")
      .setMaxIter(10)
    
val gbtparamGrid = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array(5, 30))
  .build()

val steps:Array[PipelineStage] = Array(jobIndexer,marIndexer,eduIndexer,defIndexer,housIndexer,loanIndexer,contIndexer,monIndexer,poutIndexer, yIndexer, oneHot, vectorAssembler,gbt)

val gbtPipeline = new Pipeline()
  .setStages(steps)


val gbtevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("yIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

val gbtcv = new CrossValidator() 
  .setEstimator(gbtPipeline)
  .setEvaluator(gbtevaluator)
  .setEstimatorParamMaps(gbtparamGrid)
  .setNumFolds(2)

val GBTmodel = gbtcv.fit(train)

// COMMAND ----------

val holdout1 = GBTmodel.bestModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "yIndex", 
    """CASE double(round(prediction)) = yIndex
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout1)

// COMMAND ----------

val accur1 = holdout1.selectExpr("sum(equal)/sum(1)")
display(accur1)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val rm1 = new MulticlassMetrics(
  holdout1.select("prediction", "yIndex").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("Confusion Matrix:\n" + rm1.confusionMatrix)
println("Accuracy: " + rm1.accuracy)

val labels = rm1.labels
labels.foreach { l =>
  println(s"Precision($l) = " + rm1.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + rm1.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + rm1.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + rm1.fMeasure(l))
}

// Weighted stats
println("Weighted precision: " + rm1.weightedPrecision)
println("Weighted recall: " + rm1.weightedRecall)
println(s"Weighted F1 score: " + rm1.weightedFMeasure)
println("Weighted false positive rate: " + rm1.weightedFalsePositiveRate)

// COMMAND ----------

// MAGIC %md ## Random Forest

// COMMAND ----------

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
//import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
//import org.apache.spark.ml.{Pipeline, PipelineStage}


val rfModel = new RandomForestClassifier()
  .setLabelCol("yIndex")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(rfModel.maxDepth, Array(25, 30))
  .addGrid(rfModel.numTrees, Array(80, 110))
  .build()

val steps:Array[PipelineStage] = Array(jobIndexer,marIndexer,eduIndexer,defIndexer,housIndexer,loanIndexer,contIndexer,monIndexer,poutIndexer, yIndexer, oneHot, vectorAssembler,rfModel)

val rfPipeline = new Pipeline()
  .setStages(steps)

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("yIndex")

val cv = new CrossValidator() 
  .setEstimator(rfPipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(2)

val pipelineFitted = cv.fit(train)


// COMMAND ----------

val holdout2 = pipelineFitted.bestModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "yIndex", 
    """CASE double(round(prediction)) = yIndex
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout2)

// COMMAND ----------

val accur2 = holdout2.selectExpr("sum(equal)/sum(1)")
display(accur2)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val rm2 = new MulticlassMetrics(
  holdout2.select("prediction", "yIndex").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("Confusion Matrix:\n" + rm2.confusionMatrix)
println("Accuracy: " + rm2.accuracy)

val labels = rm2.labels
labels.foreach { l =>
  println(s"Precision($l) = " + rm2.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + rm2.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + rm2.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + rm2.fMeasure(l))
}

// Weighted stats
println("Weighted precision: " + rm2.weightedPrecision)
println("Weighted recall: " + rm2.weightedRecall)
println("Weighted F1 score: " + rm2.weightedFMeasure)
println("Weighted false positive rate: " + rm2.weightedFalsePositiveRate)

// COMMAND ----------

import spark.implicits._
val metricsDF2= Seq(
  ("Precision(0.0)",rm.precision(0),rm3.precision(0),rm1.precision(0),rm2.precision(0)),
  ("Precision(1.0)",rm.precision(1),rm3.precision(1),rm1.precision(1),rm2.precision(1)),
  ("Recall(0.0)",rm.recall(0),rm3.recall(0),rm1.recall(0),rm2.recall(0)),
  ("Recall(1.0)",rm.recall(1),rm3.recall(1),rm1.recall(1),rm2.recall(1)),
  //("FPR(0.0)",rm.falsePositiveRate(0),rm3.falsePositiveRate(0),rm1.falsePositiveRate(0),rm2.falsePositiveRate(0)),
  //("FPR(1.0)",rm.falsePositiveRate(1),rm3.falsePositiveRate(1),rm1.falsePositiveRate(1),rm2.falsePositiveRate(1)),
  ("F1-Score(0.0)",rm.fMeasure(0),rm3.fMeasure(0),rm1.fMeasure(0),rm2.fMeasure(0)),
  ("F1-Score(1.0)",rm.fMeasure(1),rm3.fMeasure(1), rm1.fMeasure(1),rm2.fMeasure(1))
).toDF("Metrics", "Logistic Regression","Linear SVC","GBT Classifier","Random Forest")

display(metricsDF2)

// COMMAND ----------

val metricsDF3= Seq(
  ("Weighted F1 score", rm.weightedFMeasure,rm3.weightedFMeasure,rm1.weightedFMeasure,rm2.weightedFMeasure),
  ("Weighted precision",rm.weightedPrecision,rm3.weightedPrecision,rm1.weightedPrecision,rm2.weightedPrecision),
  ("Weighted recall",rm.weightedRecall,rm3.weightedRecall,rm1.weightedRecall,rm2.weightedRecall)
).toDF("Metrics", "Logistic Regression","Linear SVC","GBT Classifier","Random Forest")

display(metricsDF3)
