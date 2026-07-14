package imageprocessing

import java.io.File
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import zio.*
import zio.Console.*
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.{Sequential, Dense, ReLU, Output, Dropout, CategoricalCrossEntropy, SGD}

object MnistClassifier extends ZIOAppDefault:

  private val height = 28
  private val width = 28
  private val outputNum = 10
  private val numEpochs = 15

  override def run: ZIO[Any, Throwable, Unit] =
    ZIO.scoped {
      for {
        path <- DataUtilities.dataPath
        _ <- DataUtilities.downloadData

        trainDir = File(s"$path/mnist_png/training")
        testDir = File(s"$path/mnist_png/testing")

        _ <- printLine("Loading training data...")
        (trainImages, trainLabels) <- loadImages(trainDir)
        _ <- printLine(s"Loaded ${trainImages.length} training samples")

        _ <- printLine("Loading test data...")
        (testImages, testLabels) <- loadImages(testDir)
        _ <- printLine(s"Loaded ${testImages.length} test samples")

        _ <- printLine("BUILD MODEL")
        arch = Sequential(height * width)(
          Dense(100, ReLU),
          Output(outputNum, CategoricalCrossEntropy(1e-15))
        ).withOptimizer(SGD(0.006)).withSeed(123).build

        model <- create(arch)

        _ <- printLine("TRAIN MODEL")
        result <- model.fitZ(trainImages, trainLabels, numEpochs, 0.006f)
        _ <- printLine(s"Training complete, final loss: ${result.loss}")

        _ <- printLine("EVALUATE MODEL")
        metrics <- model.evaluateZ(testImages, testLabels, List(EvalMetric.Accuracy))
        accuracy = metrics("accuracy")
        _ <- printLine(f"Accuracy: $accuracy%.4f (${accuracy * 100}%.2f%%)")
      } yield ()
    }

  private def loadImages(dir: File): Task[(Array[Array[Float]], Array[Int])] =
    ZIO.attemptBlocking {
      val images = ArrayBuffer[Array[Float]]()
      val labels = ArrayBuffer[Int]()
      for (labelDir <- dir.listFiles().filter(_.isDirectory).sortBy(_.getName.toInt)) {
        val label = labelDir.getName.toInt
        for (file <- labelDir.listFiles().filter(_.getName.endsWith(".png"))) {
          val img = ImageIO.read(file)
          val pixels = new Array[Float](height * width)
          var y = 0
          while y < height do
            var x = 0
            while x < width do
              pixels(y * width + x) = (img.getRGB(x, y) & 0xFF) / 255.0f
              x += 1
            y += 1
          images += pixels
          labels += label
        }
      }
      (images.toArray, labels.toArray)
    }
