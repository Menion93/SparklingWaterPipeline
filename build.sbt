import sbt.project

name := "h20try"
version := "1.0"
scalaVersion := "2.11.5"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "1.5.1",
  "ai.h2o"%"sparkling-water-core_2.11"%"2.1.9",
  "ai.h2o"%"sparkling-water-ml_2.11"%"2.1.9")

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}