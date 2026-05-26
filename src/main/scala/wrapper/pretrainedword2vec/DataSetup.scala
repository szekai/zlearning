package wrapper.pretrainedword2vec

import org.apache.commons.io.FileUtils
import zio.*

import java.io.File
import java.net.URL
import java.nio.file.Path

object DataSetup:

  // Custom error ADT
  sealed trait SetupError extends Throwable
  case class DownloadError(cause: Throwable) extends SetupError
  case class ModelLoadError(cause: Throwable) extends SetupError

  // ZIO-version of data download
  def downloadDataset: ZIO[Any, SetupError, Unit] = ZIO.scoped {
    val dataUrl = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    val dataPath = sys.props("java.io.tmpdir") + "/dl4j_w2vSentiment/"

    ZIO.attemptBlocking {
        // Original Java download logic here
        FileUtils.copyURLToFile(new URL(dataUrl), new File(dataPath))
      }.mapError(DownloadError.apply)
      .ensuring(ZIO.attempt(FileUtils.cleanDirectory(new File(dataPath))).ignore)
  }

  /**
   * Load Word2Vec vectors using zio-nn 0.8.0 embedding API.
   * Returns a ZIO-native Word2Vec model with similarity/embedding support.
   */
  def loadWordVectors(path: String): ZIO[Any, SetupError, Word2VecModel] =
    Word2VecLoader.loadGoogleNewsVectors(path)
      .mapError(e => ModelLoadError(e))

  /**
   * Legacy: load raw DL4J WordVectors for consumers that need the native type
   * (e.g. DataSetIteratorWord2Vec).
   */
  def loadWordVectorsRaw(path: String): ZIO[Any, SetupError, org.deeplearning4j.models.embeddings.wordvectors.WordVectors] =
    ZIO.attemptBlocking(
      org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.loadStaticModel(new File(path))
    ).mapError(ModelLoadError.apply)
