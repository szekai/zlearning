package demo

import zio.*
import zio.nn.dsl.*
import zio.nn.dl4j.zioApi.*

object HelloWorldZioNN extends ZIOAppDefault:

  private val xorFeatures: Array[Array[Float]] = Array(
    Array(0.0f, 0.0f), Array(0.0f, 1.0f),
    Array(1.0f, 0.0f), Array(1.0f, 1.0f)
  )
  private val xorLabels: Array[Float] = Array(0.0f, 1.0f, 1.0f, 0.0f)

  override def run: ZIO[Any, Throwable, Unit] =
    ZIO.scoped {
      for
        arch <- ZIO.attempt(
          Sequential(2)(
            Dense(4, ReLU),
            Dense(4, ReLU),
            Output(1, MSE)
          ).withOptimizer(SGD(0.1)).build
        )

        model <- create(arch)
        _ <- Console.printLine("Training XOR with zio-nn DSL...")
        result <- model.fitZ(xorFeatures, xorLabels, epochs = 1000, lr = 0.1f)
        _ <- Console.printLine(s"Final loss: ${result.loss} after ${result.epochs} epochs")

        _ <- Console.printLine("\nPredictions:")
        predictions <- model.predictZ(xorFeatures)
        _ <- ZIO.foreach(xorFeatures.zip(predictions)) { case (input, pred) =>
          Console.printLine(f"  [${input(0)}%.0f, ${input(1)}%.0f] -> ${pred}%.4f")
        }
      yield ()
    }
