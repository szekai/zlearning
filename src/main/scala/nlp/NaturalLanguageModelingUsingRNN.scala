package nlp

import zio._
import zio.Console._
import zio.nn.dl4j.zioApi.*
import zio.nn.dsl.{Sequential, LSTM, Dense, Output, Softmax, Tanh, CategoricalCrossEntropy}
import zio.nn.OptimizerDef
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable

object NaturalLanguageModelingUsingRNN extends ZIOAppDefault:

  private val LEARN_STRING: Array[Char] = "* he cleaning agent cleans the mailbox.".toCharArray
  private val HIDDEN_LAYER_WIDTH = 50

  def run =
    val program = ZIO.scoped {
      for
        charList <- createListOfPossibleCharacters()
        vocabSize = charList.size
        _ <- printLine(s"Characters: ${charList.mkString(" ")}")

        arch = Sequential(vocabSize)(
          LSTM(HIDDEN_LAYER_WIDTH, Tanh),
          LSTM(HIDDEN_LAYER_WIDTH, Tanh),
          Output(vocabSize, CategoricalCrossEntropy(1e-15), Softmax)
        ).withOptimizer(OptimizerDef.RMSprop(0.001)).withSeed(123).build

        model <- create(arch)

        trainingData <- createTrainingData(charList)
        _ <- ZIO.foreachDiscard(0 until 500) { epoch =>
          processEpoch(epoch, model, trainingData, charList)
        }
      yield ()
    }
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
    model: zio.nn.dl4j.ZModel,
    trainingData: org.nd4j.linalg.dataset.DataSet,
    charList: mutable.Buffer[Char]
  ): ZIO[Any, Throwable, Unit] =
    for
      _ <- printLine(s"Epoch $epoch")
      _ <- ZIO.attemptBlocking(model.underlying.fit(trainingData))
      _ <- model.rnnClearPreviousStateZ
      testInit <- ZIO.attempt(Nd4j.zeros(1, charList.size, 1))
      initialChar <- safeCharIndex(charList, LEARN_STRING(0))
      _ <- safePutScalar(testInit, Array(0, initialChar, 0), 1)
      initialOutput <- model.rnnTimeStepZ(testInit)
      _ <- predictAndPrintSequence(model, initialOutput, charList)
      _ <- printLine("")
    yield ()

  private def predictAndPrintSequence(
    model: zio.nn.dl4j.ZModel,
    initialOutput: INDArray,
    charList: mutable.Buffer[Char]
  ): ZIO[Any, Throwable, Unit] =
    ZIO.foldLeft(LEARN_STRING.indices)(initialOutput) { (currentOutput, _) =>
      for
        sampledIdx <- getHighestScoreNeuron(currentOutput)
        _ <- print(charList(sampledIdx).toString)
        nextInput <- createNextInput(sampledIdx, charList.size)
        nextOutput <- model.rnnTimeStepZ(nextInput)
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

  private def createListOfPossibleCharacters(): ZIO[Any, Throwable, mutable.Buffer[Char]] =
    ZIO.attempt {
      mutable.Buffer.from(mutable.LinkedHashSet.from(LEARN_STRING))
    }

  private def createTrainingData(
    charList: mutable.Buffer[Char]
  ): ZIO[Any, Throwable, org.nd4j.linalg.dataset.DataSet] =
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
    } yield new org.nd4j.linalg.dataset.DataSet(input, labels)
