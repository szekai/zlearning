package imageprocessing

import org.apache.commons.io.FilenameUtils
import zio.*

import java.io.*

object DataUtilities:

  val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

  // Access system property to get temporary path
  val dataPath: ZIO[Any, Throwable, String] = for {
    tmpDirOpt <- System.property("java.io.tmpdir")
    tmpDir <- ZIO.fromOption(tmpDirOpt).orElseFail(new Exception("Temporary directory not found"))
  } yield FilenameUtils.concat(tmpDir, "dl4j_Mnist/")

  def downloadData: Task[Unit] =
    for {
      path          <- dataPath
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


  def getMnistPNG(path: String): Task[Unit] =
    val archivePath = s"$path/mnist_png.tar.gz"
    for {
      file <- ZIO.attempt(new File(archivePath))
      _    <- if (!file.exists()) then
        common.DataUtilities.loadFile(DATA_URL, archivePath) *>
          ZIO.logInfo(s"Data downloaded to $archivePath")
      else
        ZIO.logInfo(s"Using existing file at ${file.getAbsolutePath}")
    } yield ()

  def downloadFile(remoteUrl: String, localPath: String): Task[Boolean] =
    common.DataUtilities.loadFile(remoteUrl, localPath)
  
  // Extract the tar.gz file
  def extractTarGz(inputPath: String, outputPath: String): Task[Unit] =
    common.DataUtilities.extractTarGz(inputPath, outputPath)
