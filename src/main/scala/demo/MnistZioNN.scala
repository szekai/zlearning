package demo

import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import zio.*
import zio.nn.dsl.*
import zio.nn.dl4j.zioApi.*
import java.io.File
import java.util.Random

object MnistZioNN extends ZIOAppDefault:
  private val height = 28; private val width = 28; private val channels = 1
  private val inputSize = height * width; private val numClasses = 10
  private val rngseed = 123; private val batchSize = 128; private val numEpochs = 5

  override def run: ZIO[Any, Throwable, Unit] =
    ZIO.scoped {
    for
      path <- imageprocessing.DataUtilities.dataPath
      _ <- imageprocessing.DataUtilities.downloadData
      trainDir = File(s"$path/mnist_png/training")
      testDir = File(s"$path/mnist_png/testing")

      _ <- Console.printLine("Loading MNIST data...")
      (trainF, trainL) <- loadMnistData(trainDir)
      (testF, testL) <- loadMnistData(testDir)
      _ <- Console.printLine(s"Train: ${trainF.length}, Test: ${testF.length}")

      arch <- ZIO.attempt(
        Sequential(inputSize)(
          Dense(100, ReLU),
          Output(numClasses, MSE)
        ).build
      )

      _ <- Console.printLine(s"Architecture: $inputSize -> 100 -> $numClasses (zio-nn DSL)")

      _ <- Console.printLine("Creating model...")
      model <- create(arch)

      _ <- Console.printLine(s"Training $numEpochs epochs...")
      _ <- model.fitZ(trainF, trainL, epochs = numEpochs, lr = 0.006f)
        .catchAll(e => Console.printLineError(s"Note: ${e.getMessage} (issue #16)"))

      _ <- Console.printLine("Predicting...")
      preds <- model.predictZ(testF)
      acc = preds.indices.count(i => preds(i).toInt == testL(i).toInt).toDouble / testF.length
      _ <- Console.printLine(f"Accuracy: ${acc * 100}%.2f%%")
    yield ()
    }

  private def loadMnistData(dataDir: File): ZIO[Any, Throwable, (Array[Array[Float]], Array[Float])] =
    ZIO.attempt {
      val randNumGen = Random(rngseed)
      val split = FileSplit(dataDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
      val labelMaker = ParentPathLabelGenerator()
      val rr = ImageRecordReader(height, width, channels, labelMaker)
      rr.initialize(split)
      val di = RecordReaderDataSetIterator(rr, batchSize, 1, numClasses)
      val s = ImagePreProcessingScaler(0, 1); s.fit(di); di.setPreProcessor(s)
      val f = scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
      val l = scala.collection.mutable.ArrayBuffer.empty[Float]
      while di.hasNext do
        val ds = di.next()
        val n = ds.getFeatures.shape()(0).toInt
        for i <- 0 until n do
          f += ds.getFeatures.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).reshape(1, inputSize).toFloatVector
          l += ds.getLabels.get(NDArrayIndex.point(i)).argMax().getInt(0).toFloat
      (f.toArray, l.toArray)
    }
