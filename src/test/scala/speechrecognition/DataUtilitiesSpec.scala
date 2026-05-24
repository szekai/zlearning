package speechrecognition

import zio.*
import zio.test.*
import zio.test.Assertion.*
import utils.TestUtils

import java.nio.file.Files

object DataUtilitiesSpec extends ZIOSpecDefault {

  val sampleText = "Hello from inside the tar!"

  override def spec = suite("DataUtilitiesSpec")(
    test("extractTarGz should unpack a tar.gz archive") {
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
    },

    test("downloadFile should return false if file already exists") {
      for {
        tmpFile <- ZIO.attempt(Files.createTempFile("test_file_exists", ".tmp"))
        result  <- DataUtilities.downloadFile("http://example.com", tmpFile.toString)
      } yield assertTrue(!result)
    }
  )
}
