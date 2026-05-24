package wrapper.pretrainedword2vec

import zio._
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import java.io.File

case class Word2VecModel(vectors: WordVectors) extends AnyVal

object Word2VecLoader {
  def load(path: String): ZIO[Any, Throwable, Word2VecModel] =
    ZIO.attemptBlocking {
      Word2VecModel(
        WordVectorSerializer.loadStaticModel(new File(path))
      )
    }.refineOrDie { case e: Exception =>
      new RuntimeException(s"Failed to load Word2Vec model: ${e.getMessage}", e)
    }
}
