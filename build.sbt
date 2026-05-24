ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.8.3"

lazy val root = (project in file("."))
  .settings(
    name := "zlearning",
    zioDeps,
    dl4jDeps,
    cudaDeps,
    testFrameworks += new TestFramework("zio.test.sbt.ZTestFramework"),
    excludeDependencies += "org.scala-lang.modules" % "scala-collection-compat_2.13"
  )

lazy val zioDeps = libraryDependencies ++= Seq(
  "dev.zio" %% "zio" % "2.1.26",
  "io.github.karimagnusson" %% "zio-path" % "2.0.1",
  "dev.zio" %% "zio-test" % "2.1.26" % Test,
  "dev.zio" %% "zio-test-sbt" % "2.1.26" % Test,
  "dev.zio" %% "zio-streams" % "2.1.26",
  "dev.zio" %% "zio-logging" % "2.5.3",
  "dev.zio" %% "zio-logging-slf4j" % "2.5.3",
  "org.apache.commons" % "commons-compress" % "1.28.0",
  "io.github.szekai" %% "zio-nn-djl" % "0.7.2"
)

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
//    "org.nd4j" % "nd4j-cuda-11.4-platform" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
)