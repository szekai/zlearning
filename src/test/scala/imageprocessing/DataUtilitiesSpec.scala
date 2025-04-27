package imageprocessing

import zio._
import zio.test._
import zio.test.Assertion._
import java.io._
import java.nio.file.{Files, Paths}
import org.apache.commons.compress.archivers.tar._
import org.apache.commons.compress.compressors.gzip._

object DataUtilitiesSpec extends ZIOSpecDefault {

  val sampleFileName = "test.txt"
  val sampleContent = "This is a test file."

  def createSampleTarGz(tarGzPath: String, fileName: String, content: String): Task[Unit] = ZIO.attemptBlocking {
    val fos = new FileOutputStream(tarGzPath)
    val gcos = new GzipCompressorOutputStream(new BufferedOutputStream(fos))
    val taos = new TarArchiveOutputStream(gcos)

    val contentBytes = content.getBytes("UTF-8")
    val entry = new TarArchiveEntry(fileName)
    entry.setSize(contentBytes.length)

    taos.putArchiveEntry(entry)
    taos.write(contentBytes)
    taos.closeArchiveEntry()
    taos.finish()
    taos.close()
  }

  override def spec = suite("DataUtilities.extractTarGz")(
//    test("should extract tar.gz into the specified directory") {
//      for {
//        tempDir       <- ZIO.attempt(Files.createTempDirectory("zio-test").toFile)
//        archivePath    = s"${tempDir.getAbsolutePath}/test.tar.gz"
//        extractToPath  = s"${tempDir.getAbsolutePath}/extracted"
//
//        _             <- createSampleTarGz(archivePath, sampleFileName, sampleContent)
//        _             <- DataUtilities.extractTarGz(archivePath, extractToPath)
//
//        extractedFile  = new File(s"$extractToPath/$sampleFileName")
//        fileExists    <- ZIO.succeed(extractedFile.exists())
//        fileContent   <- ZIO.attempt(new String(Files.readAllBytes(extractedFile.toPath), "UTF-8"))
//      } yield assert(fileExists)(isTrue) && assert(fileContent)(equalTo(sampleContent))
//    }
  )
}
