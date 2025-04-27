package classify

import zio.*
import zio.Console.*
import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.impl.LineRecordReader
import org.datavec.api.split.{FileSplit, InputSplit, InputStreamInputSplit}
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.nd4j.common.primitives.Pair

import java.io.{File, IOException}
import java.net.URI
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}
import scala.jdk.javaapi.CollectionConverters.*
import org.apache.commons.lang3.StringUtils
import scala.jdk.CollectionConverters.*

import scala.jdk.javaapi.CollectionConverters

class LineRecordReaderUserTrait(val labels: List[String]) extends LineRecordReader:

  private var names: List[String] = Nil
  private var possibleCharacters: String = ""
  var maxLengthName: Int = 0
  private var totalRecords: Int = 0
  private var iter: Iterator[String] = Iterator.empty
  private var initializationDone: Boolean = false

  def initializeZIO(split: InputSplit): ZIO[Scope, Throwable, Unit] =
    split match
      case fs: FileSplit =>
        ZIO.attemptBlocking {
          val locations = fs.locations()
          if locations != null && locations.length > 1 then
            locations
          else
            throw IOException("Invalid file locations")
        }.flatMap(processFiles).catchAll(e => ZIO.fail(e))

      case _: InputStreamInputSplit =>
        ZIO.fail(IOException("InputStreamInputSplit not supported"))

  private def processFiles(locations: Array[URI]): ZIO[Any, Throwable, Unit] =
    for
      longestNameRef <- Ref.make("")
      uniqueCharsRef <- Ref.make("")
      tempNamesRef <- Ref.make(List.empty[Pair[String, List[String]]])

      // Process each location with proper effect chaining
      _ <- ZIO.foreachDiscard(locations) { location =>
        ZIO.attemptBlocking {
          val file = File(location)
          labels.find(label => file.getName == s"$label.csv") match {
            case Some(label) =>
              val path = Paths.get(file.getAbsolutePath)
              val lines = Files.readAllLines(path, Charset.defaultCharset())
              val tempList = lines.asScala.map(_.split(",")(0)).toList
              (label, tempList)
            case None =>
              throw new IOException(s"Missing label file: ${file.getName}")
          }
        }.flatMap { case (label, tempList) =>
          // Chain all Ref updates together
          for
            // Update longest name
            _ <- longestNameRef.update(current =>
              tempList.maxByOption(_.length).fold(current)(s =>
                if s.length > current.length then s else current
              )
            )
            // Update unique characters
            _ <- uniqueCharsRef.update(_ + tempList.mkString)
            // Update temp names
            _ <- tempNamesRef.update(_ :+ Pair.of(label, tempList))
          yield ()
        }
      }

      // Now get the accumulated values
      longestName <- longestNameRef.get
      uniqueChars <- uniqueCharsRef.get
      tempNames <- tempNamesRef.get

      // Validate collected data
      _ <- ZIO.fail(IOException("No characters found in training data"))
        .when(uniqueChars.isEmpty)

      // Process unique characters (same as original)
      // After processing all files
      possibleChars <- ZIO.attempt {
        uniqueChars.distinct.sorted.mkString match
          case s if s.startsWith(" ") => " " + s.trim
          case s => s
      }
      _ <- ZIO.attempt {
        this.possibleCharacters = possibleChars
        this.initializationDone = true
      }

      // Add validation for empty data
      _ <- ZIO.fail(IOException("No valid training data found"))
        .when(tempNames.isEmpty)

      minSize <- ZIO.attempt(tempNames.map(_.getSecond.size).min)
        .orElseFail(IOException("Cannot calculate min size of empty collection"))
      balanced = tempNames.map(pair => Pair.of(pair.getFirst, pair.getSecond.take(minSize)))

      binaryNames = balanced.flatMap { pair =>
        val gender = if pair.getFirst == "M" then 1 else 0
        pair.getSecond.map(name => getBinaryString(name, gender))
      }

      // Shuffle names using ZIO Random
      shuffledNames <- Random.shuffle(binaryNames)
      resolvedNames <- ZIO.foreach(shuffledNames)(identity)

      _ <- ZIO.attempt {
        maxLengthName = longestName.length
        names = resolvedNames
        totalRecords = names.size
        iter = names.iterator
      }
    yield ()

  override def initialize(split: InputSplit): Unit =
    Unsafe.unsafe { implicit unsafe =>
      Runtime.default.unsafe.run(initializeZIO(split).provideSome[Any](Scope.default)).getOrThrow()
    }

  override def next(): java.util.List[Writable] =
    if iter.hasNext then
      val currentRecord = iter.next()
      currentRecord.split(",")
        .map(s => DoubleWritable(s.toDouble))
        .toList.asJava
    else
      throw IllegalStateException("No more elements")

  override def hasNext: Boolean =
    if iter != null then iter.hasNext
    else throw IllegalStateException("Uninitialized iterator")

  override def close(): Unit = {}

  override def setConf(conf: Configuration): Unit =
    this.conf = conf

  override def getConf: Configuration = conf

  override def reset(): Unit =
    iter = names.iterator

  private def getBinaryString(name: String, gender: Int): ZIO[Any, Throwable, String] =
    ZIO.attempt {
      require(initializationDone, "Character vocabulary not initialized")

      val binaryParts = name.map { c =>
        val index = possibleCharacters.indexOf(c)
        if index == -1 then
          throw new IOException(s"Character '$c' not in vocabulary: $possibleCharacters")
        StringUtils.leftPad(Integer.toBinaryString(index), 5, "0")
      }

      val padded = StringUtils.rightPad(binaryParts.mkString, maxLengthName * 5, "0")
        .substring(0, maxLengthName * 5)

      // Split into 5-bit features with commas
      val features = padded.grouped(5).mkString(",")
      s"$features,$gender"  // Gender as last column
    }