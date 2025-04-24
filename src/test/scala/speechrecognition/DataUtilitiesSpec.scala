package speechrecognition

import org.apache.commons.compress.archivers.tar.*
import zio.*
import zio.test.*
import zio.test.Assertion.*

import java.io.*
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPOutputStream

object DataUtilitiesSpec extends ZIOSpecDefault {

  val sampleText = "Hello from inside the tar!"

  def createSampleTarGz(target: Path): UIO[Unit] = ZIO.attemptBlockingIO {
    val fos = new FileOutputStream(target.toFile)
    val gzos = new GZIPOutputStream(fos)
    val tarOut = new TarArchiveOutputStream(gzos)
    tarOut.setLongFileMode(TarArchiveOutputStream.LONGFILE_GNU)

    val data = sampleText.getBytes("UTF-8")
    val entry = new TarArchiveEntry("hello.txt")
    entry.setSize(data.length)

    tarOut.putArchiveEntry(entry)
    tarOut.write(data)
    tarOut.closeArchiveEntry()
    tarOut.close()
  }.orDie

  override def spec = suite("DataUtilitiesSpec")(

    test("extractTarGz should unpack a tar.gz archive") {
      for {
        tmpDir <- ZIO.attempt(Files.createTempDirectory("test_extract"))
        archivePath = tmpDir.resolve("sample.tar.gz")
        outputDir = tmpDir.resolve("output")
        _ <- createSampleTarGz(archivePath)
        _ <- DataUtilities.extractTarGz(archivePath.toString, outputDir.toString)
        extractedPath = outputDir.resolve("hello.txt")
        exists <- ZIO.attempt(Files.exists(extractedPath))
        content <- ZIO.attempt(new String(Files.readAllBytes(extractedPath), "UTF-8"))
      } yield assertTrue(exists) && assertTrue(content == sampleText)
    },

    test("downloadFile should return false if file already exists") {
      for {
        tmpFile <- ZIO.attempt(Files.createTempFile("test_file_exists", ".tmp"))
        result <- DataUtilities.downloadFile("http://example.com", tmpFile.toString)
      } yield assertTrue(!result)
    }

    // Note: downloadFile tests require a test HTTP server or mocking, which is better handled with tools like WireMock or sttp stub
  )
}
