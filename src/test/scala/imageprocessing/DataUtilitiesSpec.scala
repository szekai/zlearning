package imageprocessing

import zio.*
import zio.test.*
import zio.test.Assertion.*
import utils.TestUtils

import java.nio.file.Files

object DataUtilitiesSpec extends ZIOSpecDefault {

  val sampleText = "Hello from inside the tar!"

  override def spec = suite("DataUtilities.extractTarGz")(
    test("should extract tar.gz into the specified directory") {
      for {
        tmpDir    <- ZIO.attempt(Files.createTempDirectory("test_extract"))
        archivePath = tmpDir.resolve("sample.tar.gz")
        outputDir   = tmpDir.resolve("output")
        _ <- TestUtils.createTarGz(archivePath, "hello.txt", sampleText)
        _ <- DataUtilities.extractTarGz(archivePath.toString, outputDir.toString)
        extractedPath = outputDir.resolve("hello.txt")
        exists  <- ZIO.attempt(Files.exists(extractedPath))
        content <- ZIO.attempt(new String(Files.readAllBytes(extractedPath), "UTF-8"))
      } yield assertTrue(exists, content == sampleText)
    }
  )
}
