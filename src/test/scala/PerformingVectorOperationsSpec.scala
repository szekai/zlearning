import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import zio.test.*
import zio.test.Assertion.*

object PerformingVectorOperationsSpec extends ZIOSpecDefault {

  val THREE_BY_TWO_RANDOM: INDArray = Nd4j.rand(3, 2)

  override def spec: Spec[Any, Nothing] = suite("PerformingVectorOperationsSpec")(
    test("shouldCalculateMeanOnDimensionZero") {
      val mean = Nd4j.mean(THREE_BY_TWO_RANDOM, 0)
      println(s"Mean on dimension zero: $mean")
      assertTrue(mean.shape().sameElements(Array(2)))
    },

    test("shouldCalculateSum") {
      val sum: Number = THREE_BY_TWO_RANDOM.sumNumber()
      println(s"Sum: $sum")
      assertTrue(sum.doubleValue() > 0.0)
    },

    test("shouldCalculateMinAndMax") {
      val min = THREE_BY_TWO_RANDOM.minNumber()
      val max = THREE_BY_TWO_RANDOM.maxNumber()
      println(s"Min: $min")
      println(s"Max: $max")
      assertTrue(min.doubleValue() <= max.doubleValue())
    },

    test("shouldCalculateVarianceAndStandardDeviation") {
      // Compute variance along the first dimension (rows)
      val variance = Nd4j.`var`(THREE_BY_TWO_RANDOM, 0)
      // Compute standard deviation along the second dimension (columns)
      val stdDev = Nd4j.std(THREE_BY_TWO_RANDOM, 1)

      println(s"Variance: $variance")
      println(s"Standard deviation: $stdDev")

      assertTrue(
        variance.shape().sameElements(Array(2)),  // variance should have the same shape as the columns
        stdDev.shape().sameElements(Array(3))    // standard deviation should match number of rows
      )
    }

  )
}
