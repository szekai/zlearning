package wrapper.pretrainedword2vec

import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import zio.*

case class NetworkConfig(config: MultiLayerConfiguration)

val networkConfigLayer = ZLayer.succeed {
  val seed = 0         // Seed for reproducibility
  val vectorSize = 300  // Size of the word vectors. 300 in the Google News model
  NetworkConfig(new NeuralNetConfiguration.Builder()
    .seed(seed)
    .updater(new Adam(5e-3))
    .l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
    .list()
    .layer(new LSTM.Builder().nIn(vectorSize).nOut(256)
      .activation(Activation.TANH).build())
    .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
    .build())
}
