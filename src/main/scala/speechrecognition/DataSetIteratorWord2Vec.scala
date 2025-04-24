package speechrecognition

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.File
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*
import scala.util.Try

class DataSetIteratorWord2Vec(
                               dataDirectory: String,
                               wordVectors: WordVectors,
                               batchSize: Int,
                               truncateLength: Int,
                               train: Boolean
                             ) extends DataSetIterator {

  private val vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length
  private var cursor = 0
  private val positiveFiles = new File(FilenameUtils.concat(dataDirectory, s"aclImdb/${if (train) "train" else "test"}/pos/")).listFiles()
  private val negativeFiles = new File(FilenameUtils.concat(dataDirectory, s"aclImdb/${if (train) "train" else "test"}/neg/")).listFiles()

  private val tokenizerFactory: TokenizerFactory = {
    val tf = new DefaultTokenizerFactory()
    tf.setTokenPreProcessor(new CommonPreprocessor())
    tf
  }

  override def next(num: Int): DataSet = {
    if (!hasNext) throw new NoSuchElementException("No more examples available")
    Try(nextDataSet(num)).getOrElse(throw new RuntimeException("Failed to load dataset"))
  }

  private def nextDataSet(num: Int): DataSet = {
    // Load reviews
    val reviews = new scala.collection.mutable.ArrayBuffer[String](num)
    val positive = new Array[Boolean](num)
    for (i <- 0 until num if cursor < totalExamples()) {
      val review = if (cursor % 2 == 0) {
        val posReviewNumber = cursor / 2
        positive(i) = true
        FileUtils.readFileToString(positiveFiles(posReviewNumber))
      } else {
        val negReviewNumber = cursor / 2
        positive(i) = false
        Files.readString(Paths.get(negativeFiles(negReviewNumber).getPath))
      }
      reviews += review
      cursor += 1
    }

    // Tokenize and filter
    val allTokens = reviews.map { review =>
      val tokens = tokenizerFactory.create(review).getTokens
      tokens.asScala.filter(wordVectors.hasWord)
    }
    val maxLength = allTokens.map(_.size).max
    val truncatedLength = math.min(maxLength, truncateLength)

    // Create DataSet
    val features = Nd4j.create(Array(reviews.size, vectorSize, truncatedLength), 'f')
    val labels = Nd4j.create(Array(reviews.size, 2, truncatedLength), 'f')
    val featuresMask = Nd4j.zeros(reviews.size, truncatedLength)
    val labelsMask = Nd4j.zeros(reviews.size, truncatedLength)

    for (i <- reviews.indices) {
      val tokens = allTokens(i)
      val seqLength = math.min(tokens.size, truncatedLength)
      val vectors = wordVectors.getWordVectors(tokens.take(seqLength).asJava).transpose()

      features.put(
        Array(
          NDArrayIndex.point(i),
          NDArrayIndex.all(),
          NDArrayIndex.interval(0, seqLength)
        ),
        vectors
      )
      featuresMask.get(Array(NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)):_*).assign(1)

      val idx = if (positive(i)) 0 else 1
      val lastIdx = math.min(tokens.size, truncatedLength)
      labels.putScalar(Array(i, idx, lastIdx - 1), 1.0)
      labelsMask.putScalar(Array(i, lastIdx - 1), 1.0)
    }



    new DataSet(features, labels, featuresMask, labelsMask)
  }

  def totalExamples(): Int = positiveFiles.length + negativeFiles.length

  override def inputColumns(): Int = vectorSize

  override def totalOutcomes(): Int = 2

  override def reset(): Unit = cursor = 0

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = batchSize

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit =
    throw new UnsupportedOperationException()

  override def getPreProcessor: DataSetPreProcessor =
    throw new UnsupportedOperationException()

  override def getLabels: java.util.List[String] =
    List("positive", "negative").asJava

  override def hasNext: Boolean = cursor < totalExamples()

  override def next(): DataSet = next(batchSize)

  override def remove(): Unit = ()

  def loadReviewToString(index: Int): String = {
    val file = if (index % 2 == 0) positiveFiles(index / 2) else negativeFiles(index / 2)
    FileUtils.readFileToString(file)
  }

  def isPositiveReview(index: Int): Boolean = index % 2 == 0

  // Load features from file
  def loadFeaturesFromFile(file: File, maxLength: Int): INDArray = {
    val review = Files.readString(Paths.get(file.getPath))
    transformStringIntoFeatureVectorOfNumber(review, maxLength)
  }

  // Convert review to features
  def transformStringIntoFeatureVectorOfNumber(reviewContents: String, maxLength: Int): INDArray = {
    val tokens = tokenizerFactory.create(reviewContents).getTokens.asScala
    val filteredTokens = tokens.filter(wordVectors.hasWord)
    val outputLength = math.min(maxLength, filteredTokens.size)

    val features = Nd4j.create(1, vectorSize, outputLength)
    var count = 0

    for (j <- tokens.indices if count < maxLength) {
      val token = tokens(j)
      val vector = wordVectors.getWordVectorMatrix(token)
      if (vector != null) {
        features.put(
          Array(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)),
          vector
        )
        count += 1
      }
    }

    features
  }
}

