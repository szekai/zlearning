package speechrecognition

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import speechrecognition.imdb.ImdbDataDownloader
import zio.{System, *}

import java.io.File
import java.nio.charset.StandardCharsets

object HumanSpeechRecognitionUsingClassification extends ZIOAppDefault {

  val BATCH_SIZE = 64
  val SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL = 300
  val N_EPOCHS = 1
  val MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW = 256
  val SEED = 0

  val IMDB_COMMENTS_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  //download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
  val GOOGLE_NEWS_VECTOR_PATH = "/Users/szekai/Downloads/GoogleNews-vectors-negative300.bin.gz"

  val getIMDBDataPath: ZIO[Any, Throwable, String] = for {
    tmpDirOpt <- System.property("java.io.tmpdir")
    tmpDir <- ZIO.fromOption(tmpDirOpt).orElseFail(new Exception("Temporary directory not found"))
  } yield FilenameUtils.concat(tmpDir, "dl4j_w2vSentiment/")

  override def run: ZIO[Any, Throwable, Unit] = for {
    imdbPath <- getIMDBDataPath

    _ <- ImdbDataDownloader.downloadIMDBDatabase(imdbPath)

    _ <- ZIO.succeed(Nd4j.getMemoryManager.setAutoGcWindow(10000))

    net <- configureMultiLayerWithTwoOutputClasses()

    wordVectors <- ZIO.attempt(WordVectorSerializer.loadStaticModel(new File(GOOGLE_NEWS_VECTOR_PATH)))
    train = new DataSetIteratorWord2Vec(imdbPath, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, true)
    test = new DataSetIteratorWord2Vec(imdbPath, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, false)

    _ <- performTraining(net, train, test)
    _ <- printFirstPredictedPositiveReview(net, test, imdbPath)
  } yield ()

  private def printFirstPredictedPositiveReview(
                                                 net: MultiLayerNetwork,
                                                 test: DataSetIteratorWord2Vec,
                                                 imdbPath: String
                                               ): ZIO[Any, Throwable, Unit] = {
    val reviewFile = new File(FilenameUtils.concat(imdbPath, "aclImdb/test/pos/0_10.txt"))
    val zioDataSetService = new ZIODataSetService(test)
    for {
      reviewText <- ZIO.attempt(FileUtils.readFileToString(reviewFile, StandardCharsets.UTF_8))
      features <- zioDataSetService.featuresFromString(reviewText, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW)
      output <- ZIO.succeed(net.output(features))
      tsLength = output.size(2)
      probs = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(tsLength - 1))
      _ <- Console.printLine(
        s"""
           |-------------------------------
           |First positive review:
           |$reviewText
           |
           |Probabilities at last time step:
           |p(positive): ${probs.getDouble(0L)}
           |p(negative): ${probs.getDouble(1L)}
           |-------------------------------
           |""".stripMargin)
    } yield ()
  }

  private def performTraining(
                               net: MultiLayerNetwork,
                               train: DataSetIteratorWord2Vec,
                               test: DataSetIteratorWord2Vec
                             ): ZIO[Any, Throwable, Unit] = {
    for {
      _ <- Console.printLine("Starting training")
      _ <- ZIO.foreachDiscard(0 until N_EPOCHS) {
        epoch =>
          for {
            _ <- ZIO.succeed(net.fit(train))
            _ <- ZIO.succeed(train.reset())
            _ <- Console.printLine(s"Epoch $epoch complete. Starting evaluation:")
            eval <- ZIO.succeed {
              val evaluation: Evaluation = net.evaluate(test)
              evaluation
            }
            _ <- Console.printLine(eval.stats())
          } yield ()
      }
    } yield ()
  }

  private def configureMultiLayerWithTwoOutputClasses(): ZIO[Any, Throwable, MultiLayerNetwork] = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(SEED)
      .updater(new Adam(5e-3))
      .l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
      .list()
      .layer(0, new LSTM.Builder().nIn(SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL).nOut(256).activation(Activation.TANH).build())
      .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))
    ZIO.succeed(net)
  }
}
