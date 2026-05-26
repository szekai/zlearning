package demo

import zio.*
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
        arch <- ZIO.attempt {
          val l1 = zio.nn.LayerDef.Dense(2, 4, zio.nn.ActivationFn.ReLU)
          val l2 = zio.nn.LayerDef.Dense(4, 4, zio.nn.ActivationFn.ReLU)
          val l3 = zio.nn.LayerDef.Output(4, 1, zio.nn.LossFn.MSE, zio.nn.ActivationFn.Sigmoid)
          zio.nn.ModelDef.Sequential(zio.nn.SequentialDef(2, List(l1, l2, l3)))
        }

        _ <- Console.printLine("Creating model with zio-nn DSL...")
        model <- create(arch)

        _ <- Console.printLine("Training XOR...")
        result <- model.fitZ(xorFeatures, xorLabels, epochs = 1000, lr = 0.1f)
        _ <- Console.printLine(s"Final loss: ${result.loss} after ${result.epochs} epochs")

        _ <- Console.printLine("\nPredictions:")
        predictions <- model.predictZ(xorFeatures)
        _ <- ZIO.foreach(xorFeatures.zip(predictions)) { case (input, pred) =>
          Console.printLine(f"  [${input(0)}%.0f, ${input(1)}%.0f] -> ${pred}%.4f (expected ${input(0).toInt ^ input(1).toInt}%.0f)")
        }
      yield ()
    }
