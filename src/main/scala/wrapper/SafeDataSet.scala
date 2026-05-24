package wrapper

import zio.*

/** A safe wrapper around feature/label data as 2D float arrays.
  * Compatible with zio-nn-djl which uses Array[Array[Float]]
  * for model training and prediction.
  *
  * Each inner array represents one sample:
  * - features: one sample's feature vector
  * - labels: one sample's one-hot encoded label (or regression target)
  */
case class SafeDataSet(features: Array[Array[Float]], labels: Array[Array[Float]])

object SafeDataSet {
  def make(features: Array[Array[Float]], labels: Array[Array[Float]]): SafeDataSet =
    SafeDataSet(features, labels)
}
