package wrapper.pretrainedword2vec

import zio.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import zio.nn.dl4j.ZModel
import zio.nn.dl4j.zioApi.EvalMetric

case class Trainer(model: ZModel) {
  def trainEpoch(data: ReviewDataSet): ZIO[Any, Throwable, Unit] =
    ZIO.attemptBlocking {
      model.underlying.fit(data.features, data.labels)
    }.refineOrDie { case e: Exception =>
      new RuntimeException(s"Training failed: ${e.getMessage}", e)
    }

  def evaluate(data: ReviewDataSet): ZIO[Any, Throwable, Map[String, Double]] =
    model.evaluateZ(data.features, data.labels, List(EvalMetric.Accuracy, EvalMetric.Precision, EvalMetric.Recall, EvalMetric.F1))
}

object Trainer {
  def make(config: NetworkConfig): ZIO[Scope, Throwable, Trainer] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(new MultiLayerNetwork(config.config))
    )(model =>
      ZIO.attemptBlocking(model.close()).orDie
    ).map(m => Trainer(ZModel(m)))
}
