package speechrecognition

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.nd4j.linalg.dataset.DataSet
import speechrecognition.DataSetIteratorWord2Vec
import zio.{Task, UIO, ZIO, ZLayer}

import java.io.File

class ZIODataSetService(iterator: DataSetIteratorWord2Vec):

  /**
   * Gets the next DataSet wrapped in a ZIO Task
   */
  def nextBatch(batchSize: Int): Task[DataSet] =
    ZIO.attempt(iterator.next(batchSize))

  /**
   * Returns whether there's more data available
   */
  def hasNext: UIO[Boolean] =
    ZIO.succeed(iterator.hasNext)

  /**
   * Resets the iterator (ZIO-safe)
   */
  def reset: UIO[Unit] =
    ZIO.succeed(iterator.reset())

  /**
   * Get total number of examples in this dataset
   */
  def totalExamples: UIO[Int] =
    ZIO.succeed(iterator.totalExamples())

  /**
   * Load a review to String by index
   */
  def reviewText(index: Int): Task[String] =
    ZIO.attempt(iterator.loadReviewToString(index))

  /**
   * Whether review at index is positive
   */
  def isPositive(index: Int): UIO[Boolean] =
    ZIO.succeed(iterator.isPositiveReview(index))

  /**
   * Converts review file to INDArray feature vector
   */
  def featuresFromFile(file: File, maxLength: Int): Task[org.nd4j.linalg.api.ndarray.INDArray] =
    ZIO.attempt(iterator.loadFeaturesFromFile(file, maxLength))

  /**
   * Converts review string to INDArray feature vector
   */
  def featuresFromString(review: String, maxLength: Int): Task[org.nd4j.linalg.api.ndarray.INDArray] =
    ZIO.attempt(iterator.transformStringIntoFeatureVectorOfNumber(review, maxLength))

object ZIODataSetService:
  def layer(
             dataDirectory: String,
             wordVectors: WordVectors,
             batchSize: Int,
             truncateLength: Int,
             train: Boolean
           ): ZLayer[Any, Throwable, ZIODataSetService] =
    ZLayer.fromZIO {
      ZIO.attempt {
        val iterator = new DataSetIteratorWord2Vec(dataDirectory, wordVectors, batchSize, truncateLength, train)
        new ZIODataSetService(iterator)
      }
    }
