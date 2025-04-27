package classify

import zio.*
import zio.Console.*
import zio.Random
import org.datavec.api.split.FileSplit
import org.deeplearning4j.core.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.evaluation.classification.Evaluation
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize

import java.io.{File, IOException}

object ClassifyAndPredictTraitUsingDL extends ZIOAppDefault:

  private val Labels = List("M", "F")

  def run =
    val program = for
      resourceUrl <- ZIO.attempt(getClass.getClassLoader.getResource("Data"))
        .flatMap(url =>
          if url == null then
            ZIO.fail(new IllegalArgumentException("Resource 'Data' not found on the classpath"))
          else ZIO.succeed(url)
        )
      filePath = File(resourceUrl.toURI).getAbsolutePath
      _ <- printLine(s"Data path: $filePath")
      _ <- Classifier(filePath, Labels).startTraining()
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

        // Initialize record reader with ZIO resource management
        rr <- ZIO.acquireRelease(
          // Use the labels passed to the Classifier
          ZIO.attempt(new LineRecordReaderUserTrait(this.labels))
        )(rr => ZIO.attempt(rr.close()).ignore)

        // Use ZIO-powered initialization
        _ <- rr.initializeZIO(FileSplit(File(filePath)))

        // Get calculated parameters
        numInputs = rr.maxLengthName * 5
        numOutputs = 2
        numHiddenNodes = 2 * numInputs + numOutputs



        _ <- ZIO.attempt(rr.reset())  // Reset reader for test data

        // Configure network
        conf <- configureNeuralNetwork(seed, learningRate, numInputs, numOutputs, numHiddenNodes)
        model <- ZIO.attempt(new MultiLayerNetwork(conf)).tap(mlp => ZIO.attempt(mlp.init()))

        // Configure UI with resource cleanup
        _ <- configureUIServer(model)

        // Create and initialize dataset iterator
        trainIter <- createDataSetIterator(rr, batchSize, numInputs, numOutputs)

        // Validate data format
        _ <- validateData(rr)

        // Initialize normalizer with training data
        _ <- ZIO.attempt {
          val normalizer = trainIter.getPreProcessor.asInstanceOf[NormalizerStandardize]
          normalizer.fit(trainIter)  // Calculate mean/std
          trainIter.reset()
        }

        // Training output
        _ <- printLine("Training data format validated successfully")
//        _ <- printLine(s"Training with ${rr.totalRecords} records")
        _ <- printLine(s"Feature vector size: $numInputs")
//        _ <- printLine(s"Possible characters: ${rr.possibleCharacters}")

        // Training process
        _ <- printLine("Starting training...") *>
          performTraining(nEpochs, trainIter, model)

        // Cross-validation
        testIter <- createDataSetIterator(rr, batchSize, numInputs, numOutputs)
        _ <- printLine("Starting validation...") *>
          performCrossValidation(numOutputs, testIter, model)
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
        val actualFeatures = record.size() - 1 // Exclude label

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
                                     numInputs: Int,
                                     numOutputs: Int
                                   ): ZIO[Scope, Throwable, DataSetIterator] =
    ZIO.attempt {
      val iter = RecordReaderDataSetIterator(
        rr,
        batchSize,
        numInputs,  // Label index = last column (gender)
        numOutputs,
        true        // Classification
      )
      iter.setPreProcessor(new NormalizerStandardize())
      iter
    }

  private def configureNeuralNetwork(
                                      seed: Int,
                                      learningRate: Double,
                                      numInputs: Int,
                                      numOutputs: Int,
                                      numHiddenNodes: Int
                                    ): ZIO[Any, Throwable, MultiLayerConfiguration] =
    ZIO.attempt(
      NeuralNetConfiguration.Builder()
        .seed(seed)
        .updater(Nesterovs(learningRate, 0.9))
        .list()
        .layer(0, DenseLayer.Builder()
          .nIn(numInputs)
          .nOut(numHiddenNodes)
          .activation(Activation.RELU)
          .build())
        .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nIn(numHiddenNodes)
          .nOut(numOutputs)
          .activation(Activation.SOFTMAX)
          .build())
        .build()
    )

  private def configureUIServer(model: MultiLayerNetwork): ZIO[Scope, Throwable, Unit] =
    ZIO.acquireRelease(
      ZIO.attempt {
        val uiServer = UIServer.getInstance()
        val statsStorage = InMemoryStatsStorage()
        uiServer.attach(statsStorage)
        model.setListeners(new StatsListener(statsStorage))
        statsStorage
      }
    )(statsStorage => ZIO.attempt(statsStorage.close()).ignore).unit

  private def performTraining(
                               nEpochs: Int,
                               trainIter: DataSetIterator,
                               model: MultiLayerNetwork
                             ): ZIO[Any, Throwable, Unit] =
    ZIO.foreachDiscard(1 to nEpochs) { epoch =>
      ZIO.attemptBlocking {
        while trainIter.hasNext do model.fit(trainIter.next())
        trainIter.reset()
      } *> printLine(s"Completed epoch $epoch/${nEpochs}")
    }

  private def performCrossValidation(
                                      numOutputs: Int,
                                      testIter: DataSetIterator,
                                      model: MultiLayerNetwork
                                    ): ZIO[Any, Throwable, Unit] =
    for
      _ <- printLine("Evaluating model...")
      eval <- ZIO.attempt(new Evaluation(numOutputs))
      _ <- ZIO.attemptBlocking {
        while testIter.hasNext do
          val ds = testIter.next()
          val output = model.output(ds.getFeatures, false)
          eval.eval(ds.getLabels, output)
      }
      _ <- printLine(eval.stats())
    yield ()