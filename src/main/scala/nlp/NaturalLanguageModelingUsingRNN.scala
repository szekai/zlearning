package nlp

import zio._
import zio.Console._
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.collection.mutable

object NaturalLanguageModelingUsingRNN extends ZIOAppDefault:

  private val LEARN_STRING: Array[Char] = "* he cleaning agent cleans the mailbox.".toCharArray
  private val HIDDEN_LAYER_WIDTH = 50
  private val HIDDEN_LAYER_CONT = 2

  def run =
    val program = for
      (learnStringChars, learnStringCharsList) <- createListOfPossibleCharacters()
      builder <- parametrizeNeuralNetwork()
      listBuilder <- ZIO.attempt(builder.list())
      _ <- buildRNN(learnStringChars, listBuilder)
      outputLayerBuilder <- normalizeOutputOfNeurons(learnStringChars)
      _ <- ZIO.attempt(listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build()))
      net <- initializeNetwork(listBuilder)
      trainingData <- createTrainingData(learnStringCharsList)
      _ <- ZIO.foreachDiscard(0 until 500)(processEpoch(_, net, trainingData, learnStringCharsList))
    yield ()

    program.catchAll(e => printLineError(s"Error: ${e.getMessage}"))

  private def safePutScalar(
                             array: INDArray,
                             indices: Array[Int],
                             value: Double
                           ): ZIO[Any, Throwable, Unit] =
    ZIO.attempt {
      if indices.length != array.rank() then
        throw new IllegalArgumentException(
          s"INDArray rank ${array.rank()} doesn't match indices length ${indices.length}"
        )
      array.putScalar(indices, value)
    }

  private def safeCharIndex(
                             charList: mutable.Buffer[Char],
                             char: Char
                           ): ZIO[Any, Throwable, Int] =
    ZIO.attempt {
      val index = charList.indexOf(char)
      if index == -1 then
        throw new NoSuchElementException(s"Character '$char' not found in vocabulary")
      else index
    }

  private def processEpoch(
                            epoch: Int,
                            net: MultiLayerNetwork,
                            trainingData: DataSet,
                            charList: mutable.Buffer[Char]
                          ): ZIO[Any, Throwable, Unit] =
    for
      _ <- printLine(s"Epoch $epoch")
      _ <- ZIO.attemptBlocking(net.fit(trainingData))
      _ <- ZIO.attempt(net.rnnClearPreviousState())
      testInit <- ZIO.attempt(Nd4j.zeros(1, charList.size, 1))
      initialChar <- safeCharIndex(charList, LEARN_STRING(0))
      _ <- safePutScalar(testInit, Array(0, initialChar, 0), 1)
      initialOutput <- predictWhatShouldBeNext(net, testInit)
      _ <- predictAndPrintSequence(net, initialOutput, charList)
      _ <- printLine("")
    yield ()

  private def predictAndPrintSequence(
                                       net: MultiLayerNetwork,
                                       initialOutput: INDArray,
                                       charList: mutable.Buffer[Char]
                                     ): ZIO[Any, Throwable, Unit] =
    ZIO.foldLeft(LEARN_STRING.indices)(initialOutput) { (currentOutput, _) =>
      for
        sampledIdx <- getHighestScoreNeuron(currentOutput)
        _ <- print(charList(sampledIdx).toString)
        nextInput <- createNextInput(sampledIdx, charList.size)
        nextOutput <- predictWhatShouldBeNext(net, nextInput)
      yield nextOutput
    }.unit

  private def createNextInput(
                               sampledCharacterIdx: Int,
                               vocabSize: Int
                             ): ZIO[Any, Throwable, INDArray] =
    for
      nextInput <- ZIO.attempt(Nd4j.zeros(1, vocabSize, 1))
      _ <- safePutScalar(nextInput, Array(0, sampledCharacterIdx, 0), 1)
    yield nextInput

  private def getHighestScoreNeuron(output: INDArray): ZIO[Any, Throwable, Int] =
    ZIO.attempt(Nd4j.argMax(output, 1).getInt(0))

  private def predictWhatShouldBeNext(net: MultiLayerNetwork, input: INDArray): ZIO[Any, Throwable, INDArray] =
    ZIO.attempt(net.rnnTimeStep(input))

  private def createListOfPossibleCharacters(): ZIO[Any, Throwable, (mutable.LinkedHashSet[Char], mutable.Buffer[Char])] =
    ZIO.attempt {
      val charSet = mutable.LinkedHashSet.from(LEARN_STRING)
      val charList = mutable.Buffer.from(charSet)
      (charSet, charList)
    }

  private def createTrainingData(
                                  charList: mutable.Buffer[Char]
                                ): ZIO[Any, Throwable, DataSet] =
    for {
      input <- ZIO.attempt(Nd4j.zeros(1, charList.size, LEARN_STRING.length))
      labels <- ZIO.attempt(Nd4j.zeros(1, charList.size, LEARN_STRING.length))
      _ <- ZIO.foreachDiscard(LEARN_STRING.indices) { i =>
        for {
          currentChar <- ZIO.attempt(LEARN_STRING(i))
          nextChar <- ZIO.attempt(LEARN_STRING((i + 1) % LEARN_STRING.length))
          currentIndex <- safeCharIndex(charList, currentChar)
          nextIndex <- safeCharIndex(charList, nextChar)
          _ <- safePutScalar(input, Array(0, currentIndex, i), 1)
          _ <- safePutScalar(labels, Array(0, nextIndex, i), 1)
        } yield ()
      }
    } yield new DataSet(input, labels)

  private def initializeNetwork(listBuilder: NeuralNetConfiguration.ListBuilder): ZIO[Any, Throwable, MultiLayerNetwork] =
    ZIO.attempt {
      val conf = listBuilder.build()
      val net = new MultiLayerNetwork(conf)
      net.init()
      net.setListeners(new ScoreIterationListener(1))
      net
    }

  private def normalizeOutputOfNeurons(
                                        learnStringChars: mutable.LinkedHashSet[Char]
                                      ): ZIO[Any, Throwable, RnnOutputLayer.Builder] =
    ZIO.attempt {
      val outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
      outputLayerBuilder.activation(Activation.SOFTMAX)
      outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)
      outputLayerBuilder.nOut(learnStringChars.size)
      outputLayerBuilder
    }

  private def buildRNN(
                        learnStringChars: mutable.LinkedHashSet[Char],
                        listBuilder: NeuralNetConfiguration.ListBuilder
                      ): ZIO[Any, Throwable, Unit] =
    for
      _ <- printLine(s"Characters: ${learnStringChars.mkString(" ")}")
      _ <- ZIO.foreachDiscard(0 until HIDDEN_LAYER_CONT) {
        i =>
          ZIO.attempt {
            val hiddenLayerBuilder = new LSTM.Builder()
              .nIn(if i == 0 then learnStringChars.size else HIDDEN_LAYER_WIDTH)
              .nOut(HIDDEN_LAYER_WIDTH)
              .activation(Activation.TANH)
            listBuilder.layer(i, hiddenLayerBuilder.build())
          }
      }
    yield ()

  private def parametrizeNeuralNetwork(): ZIO[Any, Throwable, NeuralNetConfiguration.Builder] =
    ZIO.attempt {
      new NeuralNetConfiguration.Builder()
        .seed(123)
        .biasInit(0)
        .miniBatch(false)
        .updater(new RmsProp(0.001))
        .weightInit(WeightInit.XAVIER)
    }