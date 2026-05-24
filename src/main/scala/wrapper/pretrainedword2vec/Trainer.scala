package wrapper.pretrainedword2vec

import zio.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation

case class Trainer(model: MultiLayerNetwork) {
  def trainEpoch(data: ReviewDataSet): ZIO[Any, Throwable, Unit] =
    ZIO.attemptBlocking {
      model.fit(data.features, data.labels)
    }.refineOrDie { case e: Exception =>
      new RuntimeException(s"Training failed: ${e.getMessage}", e)
    }

  def evaluate(data: ReviewDataSet): ZIO[Any, Throwable, Evaluation] =
    ZIO.attemptBlocking {
      val output = model.output(data.features)
      val eval = new Evaluation()
      eval.eval(data.labels, output)
      eval
    }
}

object Trainer {
  def make(config: NetworkConfig): ZIO[Scope, Throwable, Trainer] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(new MultiLayerNetwork(config.config))
    )(model =>
      ZIO.attemptBlocking(model.close()).orDie
    ).map(Trainer(_))
}
