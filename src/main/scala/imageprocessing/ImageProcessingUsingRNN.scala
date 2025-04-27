package imageprocessing

import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.evaluation.classification.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File
import java.util.Random
import zio.*
import zio.http.Client

object ImageProcessingUsingRNN extends ZIOAppDefault:

  private val height = 28
  private val width = 28
  private val channels = 1

  private val rngseed = 123
  private val randNumGen = Random(rngseed)
  private val batchSize = 128
  private val outputNum = 10
  private val numEpochs = 1

  override def run: ZIO[Any, Throwable, Unit] =
    (for {
      path <- DataUtilities.dataPath
      _ <- DataUtilities.downloadData

      trainData = File(s"$path/mnist_png/training")
      testData = File(s"$path/mnist_png/testing")

      train = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
      test = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)

      labelMaker = ParentPathLabelGenerator()
      recordReader = ImageRecordReader(height, width, channels, labelMaker)
      _ <- ZIO.attempt(recordReader.initialize(train))

      dataIter = RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)
      scaler = ImagePreProcessingScaler(0, 1)
      _ <- ZIO.attempt(scaler.fit(dataIter))
      _ = dataIter.setPreProcessor(scaler)

      model <- buildNeuralNetwork()
      _ = model.setListeners(ScoreIterationListener(10))

      _ <- Console.printLine("TRAIN MODEL")
      _ <- ZIO.foreachDiscard(0 until numEpochs)(_ => ZIO.attempt(model.fit(dataIter)))

      _ <- Console.printLine("EVALUATE MODEL")
      _ <- ZIO.attempt(recordReader.reset())

      testIter <- ZIO.attempt(validateModel(test, recordReader, scaler))
      _ <- Console.printLine(recordReader.getLabels.toString)

      eval = Evaluation(outputNum)
      _ <- ZIO.attempt {
        while testIter.hasNext do
          val next = testIter.next()
          val output = model.output(next.getFeatures)
          eval.eval(next.getLabels, output)
      }

      _ <- Console.printLine(eval.stats())
    } yield ())
      .provide(Client.default, Scope.default)

  private def validateModel(test: FileSplit,
                            recordReader: ImageRecordReader,
                            scaler: DataNormalization): DataSetIterator =
    recordReader.initialize(test)
    val testIter = RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)
    scaler.fit(testIter)
    testIter.setPreProcessor(scaler)
    testIter

  private def buildNeuralNetwork(): ZIO[Any, Throwable, MultiLayerNetwork] =
    for
      _ <- Console.printLine("BUILD MODEL")
      conf = NeuralNetConfiguration.Builder()
        .seed(rngseed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Nesterovs(0.006, 0.9))
        .l2(1e-4)
        .list()
        .layer(0, DenseLayer.Builder()
          .nIn(height * width)
          .nOut(100)
          .activation(Activation.RELU)
          .weightInit(WeightInit.XAVIER)
          .build())
        .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nIn(100)
          .nOut(outputNum)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .build())
        .setInputType(InputType.convolutional(height, width, channels))
        .build()
      model = MultiLayerNetwork(conf)
      _ = model.init()
    yield model


  private def usingClient: Client with Scope => Client =
    identity
