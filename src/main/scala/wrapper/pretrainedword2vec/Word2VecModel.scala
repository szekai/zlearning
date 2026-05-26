package wrapper.pretrainedword2vec

import zio.*
import java.nio.file.Path

case class Word2VecModel(model: zio.nn.dl4j.embeddings.Word2VecModel)

object Word2VecLoader:
  def loadGoogleNewsVectors(path: String): ZIO[Any, Throwable, Word2VecModel] =
    ZIO.fromTry(
      zio.nn.dl4j.embeddings.Word2Vec.loadGoogleNewsVectors(Path.of(path))
    ).map(Word2VecModel(_))

  def load(path: String): ZIO[Any, Throwable, Word2VecModel] =
    ZIO.fromTry(
      zio.nn.dl4j.embeddings.Word2Vec.load(Path.of(path))
    ).map(Word2VecModel(_))

  def loadGloVe(path: String): ZIO[Any, Throwable, Word2VecModel] =
    ZIO.fail(new UnsupportedOperationException(
      "GloVe vectors return raw DL4J WordVectors. Use DataSetup.loadWordVectorsRaw for GloVe."
    ))

extension (model: Word2VecModel)
  def similarity(word1: String, word2: String): Task[Double] =
    model.model.similarity(word1, word2)

  def wordsNearest(word: String, n: Int): Task[List[String]] =
    model.model.wordsNearest(word, n)
