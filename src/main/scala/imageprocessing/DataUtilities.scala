package imageprocessing

import zio._
import zio.http._
import java.io._
import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FilenameUtils

object DataUtilities:

  val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

  // Access system property to get temporary path
  val dataPath: ZIO[Any, Throwable, String] = for {
    tmpDirOpt <- System.property("java.io.tmpdir")
    tmpDir <- ZIO.fromOption(tmpDirOpt).orElseFail(new Exception("Temporary directory not found"))
  } yield FilenameUtils.concat(tmpDir, "dl4j_Mnist/")

  // The downloadData method now needs to use Scope because we are calling methods that require Scope
  def downloadData: ZIO[Client with Scope, Throwable, Unit] =
    for {
      path          <- dataPath   // This uses System to get the tmpdir
      directory     <- ZIO.attempt(new File(path))
      _             <- ZIO.attempt(directory.mkdirs()).when(!directory.exists())
      archivePath   = s"$path/mnist_png.tar.gz"
      archiveFile   = new File(archivePath)
      extractedPath = s"${path}mnist_png"
      extractedFile = new File(extractedPath)
      _             <-
        if (!archiveFile.exists()) then
          ZIO.logInfo("Starting data download (15MB)...") *>
            getMnistPNG(path) *> extractTarGz(archivePath, path)
        else if (!extractedFile.exists()) then
          ZIO.logInfo("Data exists, extracting...") *> extractTarGz(archivePath, path)
        else
          ZIO.logInfo(s"Data already extracted at ${extractedFile.getAbsolutePath}")
    } yield ()


  // Method to get MNIST PNG, this method interacts with I/O and needs Scope for cleanup
  def getMnistPNG(path: String): ZIO[Client with Scope, Throwable, Unit] =
    val archivePath = s"$path/mnist_png.tar.gz"
    for {
      file <- ZIO.attempt(new File(archivePath))
      _    <- if (!file.exists()) then
        downloadFile(DATA_URL, archivePath) *>
          ZIO.logInfo(s"Data downloaded to $archivePath")
      else
        ZIO.logInfo(s"Using existing file at ${file.getAbsolutePath}")
    } yield ()


  // Download method that requires Client and Scope (handles file download via Client)
//  def downloadFile(remoteUrl: String, localPath: String): ZIO[Client with Scope, Throwable, Unit] =
//    for {
//      client <- ZIO.service[Client]  // Access the Client from the environment
//      req    <- ZIO.fromEither(URL.decode(remoteUrl)).map(Request.get)
//      res    <- client.request(req)
//      body   <- res.body.asStream.runCollect
//      _      <- ZIO.attemptBlocking {
//        val file = new FileOutputStream(localPath)
//        try file.write(body.toArray)
//        finally file.close()
//      }
//    } yield ()

  def downloadFile(remoteUrl: String, localPath: String, maxRedirects: Int = 5): ZIO[Client with Scope, Throwable, Unit] = {
    def download(url: String, redirectsLeft: Int): ZIO[Client with Scope, Throwable, Unit] =
      for {
        client <- ZIO.service[Client]
        req    <- ZIO.fromEither(URL.decode(url)).map(Request.get)
        res    <- client.request(req)
        _      <- if (res.status.isRedirection && redirectsLeft > 0) {
          res.header(Header.Location) match {
            case Some(location) =>
              val newUrl = location.renderedValue  // <<<<<< fixed here!
              download(newUrl, redirectsLeft - 1)
            case None =>
              ZIO.fail(new Exception("Redirected but no Location header"))
          }
        } else if (res.status.isSuccess) {
          for {
            body <- res.body.asStream.runCollect
            _    <- ZIO.attemptBlocking {
              val file = new FileOutputStream(localPath)
              try file.write(body.toArray)
              finally file.close()
            }
          } yield ()
        } else {
          ZIO.fail(new Exception(s"Unexpected status code: ${res.status}"))
        }
      } yield ()

    download(remoteUrl, maxRedirects)
  }
  
  // Extract the tar.gz file
  def extractTarGz(inputPath: String, outputPathRaw: String): Task[Unit] =
    val bufferSize = 4096
    val outputPath = if (outputPathRaw.endsWith(File.separator)) outputPathRaw else outputPathRaw + File.separator

    ZIO.attemptBlocking {
      val tais = new TarArchiveInputStream(
        new GzipCompressorInputStream(
          new BufferedInputStream(new FileInputStream(inputPath))
        )
      )

      try
        var entry: TarArchiveEntry = tais.getNextEntry
        while (entry != null) {
          val outputFile = new File(outputPath + entry.getName)
          if (entry.isDirectory) outputFile.mkdirs()
          else
            val buffer = new Array[Byte](bufferSize)
            val fos = new FileOutputStream(outputFile)
            val dest = new BufferedOutputStream(fos, bufferSize)
            try
              var count = tais.read(buffer, 0, bufferSize)
              while (count != -1) {
                dest.write(buffer, 0, count)
                count = tais.read(buffer, 0, bufferSize)
              }
            finally dest.close()
          entry = tais.getNextEntry
        }
      finally tais.close()
    }
