package com.hw

/**
  * Created by andrea on 6/16/17.
  */

import org.apache.spark.{SparkConf, SparkContext, SparkFiles}
import org.apache.spark.h2o.H2OContext
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.h2o.H2OPipeline
import org.apache.spark.ml.h2o.features.{ColRemover, DatasetSplitter}
import org.apache.spark.ml.h2o.models.H2ODeepLearning
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import water.support.{SparkContextSupport, SparkSessionSupport}
import org.apache.spark.sql.SparkSession



object h20try extends SparkContextSupport with SparkSessionSupport {

  val path = "/media/andrea/F/Data/subset/allen-p/all_documents"
  val savePath = "/media/andrea/F/Data/subset/allen-p/result/rdd.txt"

  def main(args: Array[String]): Unit = {
    // Configure this application
    val conf: SparkConf = configure("Data preparation")

    // Create SparkContext to execute application on Spark cluster
    implicit val sc = sparkContext(conf) // import implicit conversions

    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    @transient implicit val h2oContext = H2OContext.getOrCreate(sc)

    implicit val sqlContext = SparkSession.builder().getOrCreate().sqlContext

    val tokenizer = new RegexTokenizer().
      setInputCol("text").
      setOutputCol("words").
      setMinTokenLength(3).
      setGaps(false).
      setPattern("[a-zA-Z]+")

    val stopWordsRemover = new StopWordsRemover().
      setInputCol(tokenizer.getOutputCol).
      setOutputCol("filtered").
      setStopWords(Array("the", "a", "", "in", "on", "at", "as", "not", "for")).
      setCaseSensitive(false)

    val hashingTF = new HashingTF().
      setNumFeatures(1 << 10).
      setInputCol(tokenizer.getOutputCol).
      setOutputCol("wordToIndex")

    val idf = new IDF().
      setMinDocFreq(4).
      setInputCol(hashingTF.getOutputCol).
      setOutputCol("tf_idf")

    val colRemover = new ColRemover().
      setKeep(true).
      setColumns(Array[String]("label", "tf_idf"))

    val splitter = new DatasetSplitter().
      setKeys(Array[String]("train.hex", "valid.hex")).
      setRatios(Array[Double](0.8)).
      setTrainKey("train.hex")

    val dl = new H2ODeepLearning().
      setEpochs(10).
      setL1(0.001).
      setL2(0.0).
      setHidden(Array[Int](200, 200)).
      setValidKey(splitter.getKeys(1)).
      setResponseColumn("label")

    val pipeline = new H2OPipeline().
      setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, colRemover, splitter))

    val data = load(path, sc)
    val model = pipeline.fit(data)

    data.toJavaRDD.saveAsTextFile(savePath);

    return 0;

  }

  def load(dataFile: String, sc : SparkContext)(implicit sqlContext: SQLContext): DataFrame = {
  //  addFiles(sc, dataFile)

    val smsSchema = StructType(Array(
      StructField("text", StringType, nullable = false)))
    val rowRDD = sc.textFile(dataFile).map(p => Row(p))
    sqlContext.createDataFrame(rowRDD, smsSchema)
  }

}
