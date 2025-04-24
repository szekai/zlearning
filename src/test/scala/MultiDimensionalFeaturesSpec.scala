
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import zio.*
import zio.test.*

object MultiDimensionalFeaturesSpec extends ZIOSpecDefault {

  private val NUMBER_OF_ROWS = 5
  private val NUMBER_OF_COLUMNS = 6
  private val shape = Array(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)

  override def spec: Spec[TestEnvironment with Scope, Any] =
    suite("MultiDimensionalFeatures")(
      test("shouldCreateArrayFromFactoryMethods") {
        for {
          allZeros <- ZIO.attempt(Nd4j.zeros(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))
          _        <- Console.printLine("Nd4j.zeros(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)\n" + allZeros)

          allOnes <- ZIO.attempt(Nd4j.ones(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))
          _       <- Console.printLine("\nNd4j.ones(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)\n" + allOnes)

          allTens <- ZIO.attempt(Nd4j.valueArrayOf(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, 10.0))
          _       <- Console.printLine("\nNd4j.valueArrayOf(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, 10.0)\n" + allTens)
        } yield assertTrue(allZeros.shape().sameElements(Array(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)))
      },

      test("shouldCreateArrayFromPrimitiveTypeArray") {
        val vectorDouble = Array(1.0, 2.0, 3.0)
        val rowVector = Nd4j.create(vectorDouble)

        for {
          _            <- Console.printLine(s"rowVector:              $rowVector")
          _            <- Console.printLine(s"rowVector.shape():      ${rowVector.shape().mkString("[", ", ", "]")}")

            columnVector = Nd4j.create(vectorDouble, Array(3, 1))
          _            <- Console.printLine(s"columnVector:           $columnVector")
          _            <- Console.printLine(s"columnVector.shape():   ${columnVector.shape().mkString("[", ", ", "]")}")

            matrixDouble = Array(
          Array(1.0, 2.0, 3.0),
          Array(4.0, 5.0, 6.0)
          )
            matrix       = Nd4j.create(matrixDouble)
          _            <- Console.printLine("\nINDArray defined from double[][]:")
          _            <- Console.printLine(matrix.toString)
        } yield assertTrue(rowVector.length() == 3)
      },

      test("shouldCreateRandomArray") {
        for {
          uniformRandom <- ZIO.attempt(Nd4j.rand(shape: _*))
          _             <- Console.printLine("\n\n\nUniform random array:\n" + uniformRandom)
          _             <- Console.printLine(s"Full precision of random value at position (0,0): ${uniformRandom.getDouble(0L, 0L)}")

          gaussian <- ZIO.attempt(Nd4j.randn(shape))
          _        <- Console.printLine("\nN(0,1) random array:\n" + gaussian)
        } yield assertTrue(uniformRandom.length() == NUMBER_OF_ROWS * NUMBER_OF_COLUMNS)
      },

      test("shouldCreateRepeatableArray") {
        val rngSeed = 12345L
        for {
          rand1 <- ZIO.attempt(Nd4j.rand(shape, rngSeed))
          rand2 <- ZIO.attempt(Nd4j.rand(shape, rngSeed))
          _     <- Console.printLine("\nUniform random arrays with same fixed seed:")
          _     <- Console.printLine(rand1.toString)
          _     <- Console.printLine(rand2.toString)
        } yield assertTrue(rand1.equals(rand2))
      },

      test("shouldCreateMoreThan3dArray") {
        for {
          threeDimArray <- ZIO.attempt(Nd4j.ones(3, 4, 5))
          fourDimArray  <- ZIO.attempt(Nd4j.ones(3, 4, 5, 6))
          fiveDimArray  <- ZIO.attempt(Nd4j.ones(3, 4, 5, 6, 7))

          _ <- Console.printLine("\n\n\nCreating INDArrays with more dimensions:")
          _ <- Console.printLine("3d array shape: " + threeDimArray.shape().mkString("[", ", ", "]"))
          _ <- Console.printLine("4d array shape: " + fourDimArray.shape().mkString("[", ", ", "]"))
          _ <- Console.printLine("5d array shape: " + fiveDimArray.shape().mkString("[", ", ", "]"))
        } yield assertTrue(threeDimArray.shape().sameElements(Array(3, 4, 5)))
      },

      test("shouldCombineArrays") {
        for {
          rowVector1 <- ZIO.succeed(Nd4j.create(Array(1.0, 2.0, 3.0)).reshape(1, 3))
          rowVector2 <- ZIO.succeed(Nd4j.create(Array(4.0, 5.0, 6.0)).reshape(1, 3))

          vStack     <- ZIO.succeed(Nd4j.vstack(rowVector1, rowVector2))
          hStack     <- ZIO.succeed(Nd4j.hstack(rowVector1, rowVector2))

          _ <- Console.printLine("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:")
          _ <- Console.printLine("vStack:\n" + vStack)
          _ <- Console.printLine("hStack:\n" + hStack)
        } yield assertTrue(
          vStack.shape().sameElements(Array(2, 3)),
          hStack.shape().sameElements(Array(1, 6))
        )
      }
    )
}
