package wrapper.pretrainedword2vec

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import zio.stream.*
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import zio.ZIO

case class ReviewDataSet(features: INDArray, labels: INDArray)

object DataLoader {
  def loadReviews(
                   tokenizerFactory: TokenizerFactory,
                   wordVectors: WordVectors,
                   batchSize: Int
                 ): ZStream[Any, Throwable, ReviewDataSet] = {
    ZStream.fromZIO(
      ZIO.attemptBlocking {
        // Original Java example's data loading logic adapted here
        new Iterator[ReviewDataSet] {
          // Implement dataset iteration logic
          override def hasNext: Boolean = ???
          override def next(): ReviewDataSet = ???
        }
      }
    ).flatMap(iterator =>
      ZStream.fromIterator(iterator)
        .mapZIO { ds =>
          ZIO.attempt {
            ReviewDataSet(ds.features, ds.labels)
          }
        }
        .buffer(batchSize)
    )
  }
}
