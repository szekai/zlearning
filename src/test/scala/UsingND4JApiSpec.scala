import org.nd4j.linalg.factory.Nd4j
import zio.test.*
import zio.test.Assertion.*

import java.util.Arrays

object UsingND4JApiSpec extends ZIOSpecDefault {

  def spec = suite("UsingND4JApiSpec")(
    test("shouldCreateMatrix") {
      val nRows = 3
      val nColumns = 4
      val myArray = Nd4j.zeros(nRows, nColumns)

      println("Basic INDArray information:")
      println("Num. Rows:          " + myArray.rows())
      println("Num. Columns:       " + myArray.columns())
      println("Num. Dimensions:    " + myArray.rank())
      println("Shape:              " + Arrays.toString(myArray.shape()))
      println("Length:             " + myArray.length())

      println("\nArray Contents:\n" + myArray)

      println()
      println("size(0) == nRows:   " + myArray.size(0))
      println("size(1) == nColumns:   " + myArray.size(1))
      println("Is a vector:        " + myArray.isVector)
      println("Is a scalar:        " + myArray.isScalar)
      println("Is a matrix:        " + myArray.isMatrix)
      println("Is a square matrix: " + myArray.isSquare)

      assertTrue(
        myArray.rows() == 3,
        myArray.columns() == 4,
        myArray.rank() == 2,
        myArray.length() == 12,
        myArray.size(0) == 3,
        myArray.size(1) == 4,
        !myArray.isVector,
        !myArray.isScalar,
        myArray.isMatrix,
        !myArray.isSquare
      )
    },

    test("shouldPopulateArray") {
      val nRows = 4
      val nColumns = 10
      val myArray = Nd4j.zeros(nRows, nColumns)

      val val0 = myArray.getDouble(0L, 1)
      println("\nValue at (0,1):     " + val0)

      val myArray2 = myArray.add(1.0)
      println("\nNew INDArray, after adding 1.0 to each entry:")
      println(myArray2)

      val myArray3 = myArray2.mul(2.0)
      println("\nNew INDArray, after multiplying each entry by 2.0:")
      println(myArray3)

      assertTrue(
        val0 == 0.0,
        myArray2.getDouble(0L, 1) == 1.0,
        myArray3.getDouble(0L, 1) == 2.0
      )
    }
  )
}
