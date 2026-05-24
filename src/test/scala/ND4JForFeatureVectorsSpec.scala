import org.nd4j.common.util.ArrayUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.*
import zio.test.*
import zio.test.Assertion.*

object ND4JForFeatureVectorsSpec extends ZIOSpecDefault {

  val SIX_BY_THREE_RANDOM: INDArray = Nd4j.rand(6, 3)
  val TWO_BY_THREE_ONES: INDArray = Nd4j.ones(2, 3)
  val SIX_BY_THREE_ADD_TEN: INDArray = Nd4j.valueArrayOf(Array(6, 3), 10.0)

  override def spec: Spec[Any, Throwable] = suite("ND4JForFeatureVectorsSpec")(

    test("subtractVectors") {
      val array = Nd4j.rand(6, 3)
      val addTen = Nd4j.valueArrayOf(Array(6, 3), 10.0)
      val afterAdd = array.add(addTen)
      val afterSubtract = afterAdd.sub(addTen)
      assertTrue(array.equalsWithEps(afterSubtract, 1e-6))
    },

    test("multiplyVectors") {
      val array = Nd4j.rand(6, 3)
      val scale = Nd4j.valueArrayOf(Array(6, 3), 2.0)
      val multiplied = array.mul(scale)
      val divided = multiplied.div(scale)
      assertTrue(array.equalsWithEps(divided, 1e-6))
    },

    test("divideVectors") {
      val array = Nd4j.rand(6, 3)
      val divisor = Nd4j.valueArrayOf(Array(6, 3), 5.0)
      val divided = array.div(divisor)
      val restored = divided.mul(divisor)
      assertTrue(array.equalsWithEps(restored, 1e-6))
    },

    test("compareVectors") {
      val areArraysEquals1 = ArrayUtil.equals(
        SIX_BY_THREE_RANDOM.data().asFloat(),
        TWO_BY_THREE_ONES.data().asDouble()
      )
      val areArraysEquals2 = SIX_BY_THREE_RANDOM.equals(SIX_BY_THREE_RANDOM)
      assertTrue(!areArraysEquals1, areArraysEquals2)
    },

    test("sqrtVectors") {
      val absValues = abs(SIX_BY_THREE_RANDOM)
      val sqrtResult = sqrt(absValues)
      val squared = sqrtResult.mul(sqrtResult)
      assertTrue(absValues.equalsWithEps(squared, 1e-6))
    },

    test("ceilFloorAndRoundVectors") {
      val values = Nd4j.rand(6, 3)
      val ceilResult  = ceil(values)
      val floorResult = floor(values)
      val roundResult = round(values)
      assertTrue(
        ceilResult.shape().sameElements(values.shape()),
        floorResult.shape().sameElements(values.shape()),
        roundResult.shape().sameElements(values.shape())
      )
    }
  )
}
