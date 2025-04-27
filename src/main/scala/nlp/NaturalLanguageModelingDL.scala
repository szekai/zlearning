package nlp

import zio._
import zio.Console._
import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}

import java.io.File
import java.util

object NaturalLanguageModelingDL extends ZIOAppDefault:

  private def failWithMessage(message: String) =
    ZIO.fail(new Exception(message))

  def run =
    val program = for
      resourceUrl <- ZIO.attempt(getClass.getClassLoader.getResource("raw_textual_data"))
        .flatMap(url =>
          if url == null then failWithMessage("Resource 'raw_textual_data' not found on the classpath")
          else ZIO.succeed(url)
        )
      file <- ZIO.attempt(File(resourceUrl.toURI))
      vocabCache <- ZIO.attempt(new AbstractCache.Builder[VocabWord]().build())
      underlyingIterator <- ZIO.attempt(new BasicLineIterator(file))
      tokenizerFactory =
        val tf = DefaultTokenizerFactory()
        tf.setTokenPreProcessor(CommonPreprocessor())
        tf
      transformer <- ZIO.attempt(
        SentenceTransformer.Builder()
          .iterator(underlyingIterator)
          .tokenizerFactory(tokenizerFactory)
          .build()
      )
      sequenceIterator <- ZIO.attempt(
        AbstractSequenceIterator.Builder[VocabWord](transformer).build()
      )
      _ <- buildVocabulary(vocabCache, sequenceIterator)
      lookupTable <- buildWeightLookupTable(vocabCache)
      _ <- ZIO.attempt(lookupTable.resetWeights(true))
      vectors <- buildModel(vocabCache, sequenceIterator, lookupTable)
      _ <- ZIO.attemptBlocking(vectors.fit())
      _ <- findSimilarities(vectors)
    yield ()

    program.catchAll(e => printLineError(e.getMessage))

  private def findSimilarities(vectors: SequenceVectors[VocabWord]): ZIO[Any, Throwable, Unit] =
    for
      sim <- ZIO.attempt(vectors.similarity("day", "night"))
      _ <- printLine(s"Day/night similarity: $sim")
      sim2 <- ZIO.attempt(vectors.similarity("behave", "kick"))
      _ <- printLine(s"behave/kick similarity: $sim2")
      words <- ZIO.attempt(vectors.wordsNearest("day", 10))
      _ <- printLine(s"Nearest words to 'day': $words")
      words2 <- ZIO.attempt(vectors.wordsNearest("kick", 10))
      _ <- printLine(s"Nearest words to 'kick': $words2")
    yield ()

  private def buildModel(
                          vocabCache: AbstractCache[VocabWord],
                          sequenceIterator: AbstractSequenceIterator[VocabWord],
                          lookupTable: WeightLookupTable[VocabWord]
                        ): ZIO[Any, Throwable, SequenceVectors[VocabWord]] =
    ZIO.attempt(
      SequenceVectors.Builder[VocabWord](VectorsConfiguration())
        .minWordFrequency(5)
        .lookupTable(lookupTable)
        .iterate(sequenceIterator)
        .vocabCache(vocabCache)
        .batchSize(250)
        .iterations(1)
        .epochs(1)
        .resetModel(false)
        .trainElementsRepresentation(true)
        .trainSequencesRepresentation(false)
        .elementsLearningAlgorithm(SkipGram[VocabWord]())
        .build()
    )

  private def buildWeightLookupTable(vocabCache: AbstractCache[VocabWord]): ZIO[Any, Throwable, WeightLookupTable[VocabWord]] =
    ZIO.attempt(
      InMemoryLookupTable.Builder[VocabWord]()
        .vectorLength(150)
        .useAdaGrad(false)
        .cache(vocabCache)
        .build()
    )

  private def buildVocabulary(
                               vocabCache: AbstractCache[VocabWord],
                               sequenceIterator: AbstractSequenceIterator[VocabWord]
                             ): ZIO[Any, Throwable, Unit] =
    ZIO.attempt {
      val constructor = VocabConstructor.Builder[VocabWord]()
        .addSource(sequenceIterator, 5)
        .setTargetVocabCache(vocabCache)
        .build()
      constructor.buildJointVocabulary(false, true)
    }