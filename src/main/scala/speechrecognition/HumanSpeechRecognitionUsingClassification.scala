package speechrecognition

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import speechrecognition.imdb.ImdbDataDownloader
import zio.*
import zio.nn.dl4j.zioApi.*
import zio.nn.dl4j.embeddings.{Word2Vec, Word2VecModel}
import zio.nn.dsl.{Sequential, LSTM, Tanh, Output, CategoricalCrossEntropy}
import zio.nn.OptimizerDef

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Path

object HumanSpeechRecognitionUsingClassification extends ZIOAppDefault:

  val BATCH_SIZE = 64
  val SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL = 300
  val N_EPOCHS = 1
  val MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW = 256
  val SEED = 0

  val IMDB_COMMENTS_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  val GOOGLE_NEWS_VECTOR_PATH = "/Users/szekai/Downloads/GoogleNews-vectors-negative300.bin.gz"

  val getIMDBDataPath: ZIO[Any, Throwable, String] = for {
    tmpDirOpt <- System.property("java.io.tmpdir")
    tmpDir <- ZIO.fromOption(tmpDirOpt).orElseFail(new Exception("Temporary directory not found"))
  } yield FilenameUtils.concat(tmpDir, "dl4j_w2vSentiment/")

  override def run: ZIO[Any, Throwable, Unit] = ZIO.scoped {
    for {
      imdbPath <- getIMDBDataPath

      _ <- ImdbDataDownloader.downloadIMDBDatabase(imdbPath)

      _ <- ZIO.attempt(Nd4j.getMemoryManager.setAutoGcWindow(10000))

      // zio-nn: similarity demo via embedding API
      w2v <- ZIO.fromTry(Word2Vec.loadGoogleNewsVectors(Path.of(GOOGLE_NEWS_VECTOR_PATH)))
      _ <- Console.printLine("\n=== Word2Vec Similarity (zio-nn) ===")
      dayNightSim <- w2v.similarity("day", "night")
      _ <- Console.printLine(s"Similarity(day,night)=$dayNightSim")
      goodBadSim <- w2v.similarity("good", "bad")
      _ <- Console.printLine(s"Similarity(good,bad)=$goodBadSim")
      nearestDay <- w2v.wordsNearest("day", 5)
      _ <- Console.printLine(s"Nearest to 'day': ${nearestDay.mkString(", ")}")
      _ <- Console.printLine("======================================\n")

      // Training pipeline: uses raw DL4J WordVectors (DataSetIteratorWord2Vec dependency)
      wordVectors <- ZIO.attempt(WordVectorSerializer.loadStaticModel(new File(GOOGLE_NEWS_VECTOR_PATH)))

      arch = Sequential(SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL)(
        LSTM(256, Tanh),
        Output(2, CategoricalCrossEntropy(1e-15))
      ).withOptimizer(OptimizerDef.Adam(5e-3)).withSeed(SEED).build

      model <- create(arch)

      train = new DataSetIteratorWord2Vec(imdbPath, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, true)
      test = new DataSetIteratorWord2Vec(imdbPath, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, false)

      _ <- performTraining(model, train, test)
      _ <- printFirstPredictedPositiveReview(model, test, imdbPath)
    } yield ()
  }

  private def printFirstPredictedPositiveReview(
    model: zio.nn.dl4j.ZModel,
    test: DataSetIteratorWord2Vec,
    imdbPath: String
  ): ZIO[Any, Throwable, Unit] =
    val reviewFile = new File(FilenameUtils.concat(imdbPath, "aclImdb/test/pos/0_10.txt"))
    val zioDataSetService = new ZIODataSetService(test)
    for {
      reviewText <- ZIO.attempt(FileUtils.readFileToString(reviewFile, StandardCharsets.UTF_8))
      features <- zioDataSetService.featuresFromString(reviewText, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW)
      output <- ZIO.attempt(model.underlying.output(features))
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

  private def performTraining(
    model: zio.nn.dl4j.ZModel,
    train: DataSetIteratorWord2Vec,
    test: DataSetIteratorWord2Vec
  ): ZIO[Any, Throwable, Unit] =
    for {
      _ <- Console.printLine("Starting training")
      _ <- ZIO.foreachDiscard(0 until N_EPOCHS) {
        epoch =>
          for {
            _ <- ZIO.attemptBlocking(model.underlying.fit(train))
            _ <- ZIO.attempt(train.reset())
            _ <- Console.printLine(s"Epoch $epoch complete. Starting evaluation:")
            eval <- ZIO.attemptBlocking(model.underlying.evaluate(test): Evaluation)
            _ <- Console.printLine(eval.stats())
          } yield ()
      }
    } yield ()
