import org.nd4j.common.util.ArrayUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.*
import zio.*
import zio.test.*
import zio.test.Assertion.*

object ND4JForFeatureVectorsSpec extends ZIOSpecDefault {

  val SIX_BY_THREE_RANDOM: INDArray = Nd4j.rand(6, 3)
  val TWO_BY_THREE_ONES: INDArray = Nd4j.ones(2, 3)
  val SIX_BY_THREE_ADD_TEN: INDArray = Nd4j.valueArrayOf(Array(6, 3), 10.0)

  override def spec: Spec[Any, Throwable] = suite("ND4JForFeatureVectorsSpec")(

    test("subtractVectors") {
      val vectorAdd = SIX_BY_THREE_RANDOM.add(SIX_BY_THREE_ADD_TEN)
      val vectorSubtract = SIX_BY_THREE_RANDOM.sub(SIX_BY_THREE_ADD_TEN)

      for {
        _ <- Console.printLine("Vector add:\n" + vectorAdd)
        _ <- Console.printLine("Vector subtract:\n" + vectorSubtract)
      } yield assertCompletes
    },

    test("multiplyVectors") {
      val vectorMultiply = SIX_BY_THREE_RANDOM.mul(SIX_BY_THREE_ADD_TEN)

      for {
        _ <- Console.printLine("Vector multiply:\n" + vectorMultiply)
      } yield assertCompletes
    },

    test("divideVectors") {
      val vectorDivide = SIX_BY_THREE_RANDOM.div(SIX_BY_THREE_ADD_TEN)

      for {
        _ <- Console.printLine("Vector divide:\n" + vectorDivide)
      } yield assertCompletes
    },

    test("compareVectors") {
      val areArraysEquals1 = vectorEquals(SIX_BY_THREE_RANDOM, TWO_BY_THREE_ONES)
      val areArraysEquals2 = vectorEquals(SIX_BY_THREE_RANDOM, SIX_BY_THREE_RANDOM)

      for {
        _ <- Console.printLine(s"Are arrays equals: 1. $areArraysEquals1, 2. $areArraysEquals2")
      } yield assertTrue(!areArraysEquals1, areArraysEquals2)
    },

    test("sqrtVectors") {
      val sqrtResult = sqrt(SIX_BY_THREE_RANDOM)

      for {
        _ <- Console.printLine("Vector square root:\n" + sqrtResult)
      } yield assertCompletes
    },

    test("ceilFloorAndRoundVectors") {
      val ceilResult  = ceil(SIX_BY_THREE_RANDOM)
      val floorResult = floor(SIX_BY_THREE_RANDOM)
      val roundResult = round(SIX_BY_THREE_RANDOM)

      for {
        _ <- Console.printLine("Vector ceil:\n" + ceilResult)
        _ <- Console.printLine("Vector floor:\n" + floorResult)
        _ <- Console.printLine("Vector round:\n" + roundResult)
      } yield assertCompletes
    }
  )

  private def vectorEquals(arr1: INDArray, arr2: INDArray): Boolean =
    ArrayUtil.equals(arr1.data().asFloat(), arr2.data().asDouble())
}
