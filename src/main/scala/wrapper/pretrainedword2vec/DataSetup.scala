package wrapper.pretrainedword2vec

import org.apache.commons.io.FileUtils
import zio.*
import zio.stream.*

import java.io.File
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors

import java.net.URL

object DataSetup {

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
      }.mapError(DownloadError)
      .ensuring(ZIO.attempt(FileUtils.cleanDirectory(new File(dataPath))).ignore)
  }

  // Safe Word2Vec model loader with ZIO
  def loadWordVectors(path: String): ZIO[Any, SetupError, WordVectors] =
    ZIO.attemptBlocking(WordVectorSerializer.loadStaticModel(new File(path)))
      .mapError(ModelLoadError)
}
