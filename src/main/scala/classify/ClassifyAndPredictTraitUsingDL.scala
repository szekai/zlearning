package classify

import zio.*
import zio.Console.*
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.{Sequential, Dense, ReLU, Output, CategoricalCrossEntropy}
import zio.nn.OptimizerDef
import org.datavec.api.split.FileSplit
import org.deeplearning4j.core.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


import java.io.{File, IOException}
import java.net.URL
import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.jar.JarFile
import scala.jdk.CollectionConverters.*

object ClassifyAndPredictTraitUsingDL extends ZIOAppDefault:

  private val Labels = List("M", "F")

  private def extractResourceDir(jarUrl: URL, prefix: String, targetDir: File): Unit =
    val jarPath = jarUrl.getPath.substring(5, jarUrl.getPath.indexOf("!"))
    val jar = new JarFile(jarPath)
    try
      jar.entries().asIterator.asScala.foreach { entry =>
        val name = entry.getName
        if name.startsWith(prefix) && !entry.isDirectory then
          val dest = new File(targetDir, name.substring(prefix.length + 1))
          dest.getParentFile.mkdirs()
          val in = jar.getInputStream(entry)
          try Files.copy(in, dest.toPath, StandardCopyOption.REPLACE_EXISTING)
          finally in.close()
      }
    finally jar.close()

  def run =
    val program = for
      resourceUrl <- ZIO.attempt(getClass.getClassLoader.getResource("Data"))
        .flatMap(url =>
          if url == null then
            ZIO.fail(new IllegalArgumentException("Resource 'Data' not found on the classpath"))
          else ZIO.succeed(url)
        )
      dataPath <- ZIO.attemptBlocking {
        val url = resourceUrl
        if url.getProtocol == "jar" then
          val tempDir = Files.createTempDirectory("zl-data").toFile
          tempDir.deleteOnExit()
          extractResourceDir(url, "Data", tempDir)
          tempDir.getAbsolutePath
        else
          File(url.toURI).getAbsolutePath
      }
      _ <- printLine(s"Data path: $dataPath")
      _ <- Classifier(dataPath, Labels).startTraining()
    yield ()

    program.catchAll(e => printLineError(s"Error: ${e.getMessage}"))


case class Classifier(filePath: String, labels: List[String]):

  def startTraining(): ZIO[Scope, Throwable, Unit] =
    ZIO.scoped {
      for
        seed <- ZIO.succeed(123456)
        learningRate <- ZIO.succeed(0.005)
        batchSize <- ZIO.succeed(100)
        nEpochs <- ZIO.succeed(10)

        rr <- ZIO.acquireRelease(
          ZIO.attempt(new LineRecordReaderUserTrait(this.labels))
        )(rr => ZIO.attempt(rr.close()).ignore)

        _ <- rr.initializeZIO(FileSplit(File(filePath)))

        numInputs = rr.maxLengthName * 5
        numOutputs = 2
        numHiddenNodes = 2 * numInputs + numOutputs

        _ <- ZIO.attempt(rr.reset())

        arch = Sequential(numInputs)(
          Dense(numHiddenNodes, ReLU),
          Output(numOutputs, CategoricalCrossEntropy(1e-15))
        ).withOptimizer(OptimizerDef.Adam(learningRate)).withSeed(seed).build

        model <- create(arch)

        // Configure UI with resource cleanup
        _ <- configureUIServer(model)

        // Create and initialize dataset iterator
        trainIter <- createDataSetIterator(rr, batchSize, numOutputs)

        // Validate data format
        _ <- validateData(rr)

        // Initialize normalizer with training data
        _ <- ZIO.attempt {
          trainIter.getPreProcessor match
            case normalizer: NormalizerStandardize =>
              normalizer.fit(trainIter)
              trainIter.reset()
            case other =>
              throw new IllegalStateException(
                s"Expected NormalizerStandardize preprocessor but got: ${if other == null then "null" else other.getClass.getName}"
              )
        }

        // Training output
        _ <- printLine("Training data format validated successfully")
        _ <- printLine(s"Feature vector size: $numInputs")

        _ <- printLine("Starting training...") *>
          performTraining(nEpochs, trainIter, model)

        testIter <- createDataSetIterator(rr, batchSize, numOutputs)
        _ <- printLine("Starting validation...") *>
          performCrossValidation(testIter, model)
      yield ()
    }

  private def validateData(
    rr: LineRecordReaderUserTrait
  ): ZIO[Any, Throwable, Unit] =
    ZIO.attempt {
      rr.reset()
      var count = 0
      while rr.hasNext do
        val record = rr.next()
        val expectedFeatures = rr.maxLengthName * 5
        val actualFeatures = record.size() - 1

        if actualFeatures != expectedFeatures then
          throw new IOException(
            s"Data validation failed at record $count: " +
              s"Expected $expectedFeatures features, got $actualFeatures"
          )
        count += 1
      rr.reset()
    }

  private def createDataSetIterator(
    rr: LineRecordReaderUserTrait,
    batchSize: Int,
    numOutputs: Int
  ): ZIO[Scope, Throwable, DataSetIterator] =
    ZIO.attempt {
      val iter = new RecordReaderDataSetIterator(rr, batchSize, 1, numOutputs, true)
      iter.setPreProcessor(new NormalizerStandardize())
      iter
    }

  private def configureUIServer(model: zio.nn.dl4j.ZModel): ZIO[Scope, Throwable, Unit] =
    ZIO.acquireRelease(
      ZIO.attempt {
        val uiServer = UIServer.getInstance()
        val statsStorage = InMemoryStatsStorage()
        uiServer.attach(statsStorage)
        model.underlying.setListeners(new StatsListener(statsStorage))
        statsStorage
      }
    )(statsStorage => ZIO.attempt(statsStorage.close()).ignore).unit

  private def performTraining(
    nEpochs: Int,
    trainIter: DataSetIterator,
    model: zio.nn.dl4j.ZModel
  ): ZIO[Any, Throwable, Unit] =
    ZIO.foreachDiscard(1 to nEpochs) { epoch =>
      model.fitZ(trainIter, 1) *> printLine(s"Completed epoch $epoch/$nEpochs")
    }

  private def performCrossValidation(
    testIter: DataSetIterator,
    model: zio.nn.dl4j.ZModel
  ): ZIO[Any, Throwable, Unit] =
    for
      _ <- printLine("Evaluating model...")
      metrics <- model.evaluateIteratorZ(testIter, List(EvalMetric.Accuracy, EvalMetric.Precision, EvalMetric.Recall, EvalMetric.F1))
      _ <- printLine(s"Accuracy: ${metrics("accuracy")}, Precision: ${metrics("precision")}, Recall: ${metrics("recall")}, F1: ${metrics("f1")}")
    yield ()
