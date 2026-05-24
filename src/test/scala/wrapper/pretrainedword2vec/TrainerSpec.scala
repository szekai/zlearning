package wrapper.pretrainedword2vec

import zio.*
import zio.test.*
import zio.test.Assertion.*
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object TrainerSpec extends ZIOSpecDefault {

  val testConfig: NetworkConfig = NetworkConfig(
    new NeuralNetConfiguration.Builder()
      .seed(42)
      .updater(new Adam(1e-3))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(10)
        .nOut(64)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder()
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .nIn(64)
        .nOut(2)
        .build())
      .build()
  )

  override def spec = suite("Trainer")(
    test("acquireRelease should create and release a MultiLayerNetwork") {
      ZIO.scoped {
        Trainer.make(testConfig).flatMap { trainer =>
          ZIO.attempt(trainer.model.init()) *>
            ZIO.attempt(trainer.model.params())
        }
      }.map(params => assertTrue(params != null, params.length() > 0))
    },

    test("Trainer.model should be a valid MultiLayerNetwork instance") {
      ZIO.scoped {
        Trainer.make(testConfig).flatMap { trainer =>
          ZIO.attempt { trainer.model.init(); trainer.model }
        }
      }.map(model => assertTrue(model.isInstanceOf[org.deeplearning4j.nn.multilayer.MultiLayerNetwork]))
    }
  )
}
