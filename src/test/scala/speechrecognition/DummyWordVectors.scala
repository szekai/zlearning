package speechrecognition

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.embeddings.reader.ModelUtils
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.{lang, util}

class DummyWordVectors(vectorSize: Int) extends WordVectors {

  private val vocabCache: VocabCache[VocabWord] = {
    val cache = new AbstractCache[VocabWord]()
    val words = Seq("this", "is", "a", "positive", "review", "negative")
    words.foreach { word =>
      val vocabWord = new VocabWord(1.0, word)
      cache.addToken(vocabWord)
      cache.addWordToIndex(cache.numWords(), word)
      cache.putVocabWord(word)
    }
    cache
  }

  override def getWordVector(word: String): Array[Double] =
    Array.fill(vectorSize)(1.0)

  override def hasWord(word: String): Boolean = true

  override def getWordVectorMatrix(word: String): INDArray =
    Nd4j.create(getWordVector(word))

  override def getWordVectors(words: util.List[String]): INDArray =
    Nd4j.create(words.size(), vectorSize).assign(1.0)

  override def getWordVectors(words: util.Collection[String]): INDArray =
    Nd4j.create(words.size(), vectorSize).assign(1.0)

  override def vocab(): VocabCache[VocabWord] = vocabCache

  // You can stub or leave other irrelevant methods as unimplemented

  override def getUNK: String = ???

  override def setUNK(newUNK: String): Unit = ???

  override def wordsNearest(words: INDArray, top: Int): util.Collection[String] = ???

  override def wordsNearestSum(words: INDArray, top: Int): util.Collection[String] = ???

  override def wordsNearestSum(word: String, n: Int): util.Collection[String] = ???

  override def wordsNearestSum(positive: util.Collection[String], negative: util.Collection[String], top: Int): util.Collection[String] = ???

  override def accuracy(questions: util.List[String]): util.Map[String, lang.Double] = ???

  override def indexOf(word: String): Int = ???

  override def similarWordsInVocabTo(word: String, accuracy: Double): util.List[String] = ???

  override def getWordVectorMatrixNormalized(word: String): INDArray = ???

  override def getWordVectorsMean(labels: util.Collection[String]): INDArray = ???

  override def wordsNearest(positive: util.Collection[String], negative: util.Collection[String], top: Int): util.Collection[String] = ???

  override def wordsNearest(word: String, n: Int): util.Collection[String] = ???

  override def similarity(word: String, word2: String): Double = ???

  override def lookupTable(): WeightLookupTable[_ <: SequenceElement] = ???

  override def setModelUtils(utils: ModelUtils[_ <: SequenceElement]): Unit = ???

  override def outOfVocabularySupported(): Boolean = ???

  override def loadWeightsInto(array: INDArray): Unit = ???

  override def vocabSize(): Long = ???

  override def jsonSerializable(): Boolean = ???
}