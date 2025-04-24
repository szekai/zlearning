ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.6.4"

lazy val root = (project in file("."))
  .settings(
    name := "zlearning",
    zioDeps,
    sparkDeps,
    dl4jDeps,
    cudaDeps,
    testFrameworks += new TestFramework("zio.test.sbt.ZTestFramework")
  )

lazy val zioDeps = libraryDependencies ++= Seq(
  "dev.zio" %% "zio" % "2.1.17",
  //      "io.github.karimagnusson" %% "zio-path" % "2.0.1",
  "dev.zio" %% "zio-test" % "2.1.17" % Test
)

val sparkVersion = "3.5.5"

lazy val sparkDeps = libraryDependencies ++= Seq(
  ("org.apache.spark" %% "spark-core" % sparkVersion),
  ("org.apache.spark" %% "spark-sql" % sparkVersion),
  ("org.apache.spark" %% "spark-streaming" % sparkVersion),
  ("org.apache.spark" %% "spark-streaming-kafka-0-10" % sparkVersion),
  ("org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion)
).map(_.cross(CrossVersion.for3Use2_13))

val dl4j_version = "1.0.0-M2.1"
// Deeplearning4j dependencies
lazy val dl4jDeps = libraryDependencies ++= Seq(
  //  "org.deeplearning4j" %% "dl4j-spark" % dl4j_version,
  "org.deeplearning4j" % "deeplearning4j-ui" % dl4j_version,
  "org.deeplearning4j" % "deeplearning4j-parallel-wrapper" % dl4j_version
)

// Not yet support macos arm
lazy val cudaDeps = libraryDependencies ++= Seq(
  //  "org.nd4j" % "nd4j-cuda-10.0-platform" % "1.0.0-beta7",
  //  "org.nd4j" % "nd4j-cuda-9.2-platform" % "1.0.0-beta6",
  //  "org.nd4j" % "nd4j-cuda-10.1-platform" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
)