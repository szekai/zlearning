package speechrecognition

import common.DataUtilities as DataUtils
import zio.*

import java.io.IOException
import java.nio.file.{Files, Paths}

object DataUtilities:

  /** Downloads a file. Delegates to common.DataUtilities.loadFile.
    * Returns true if downloaded, false if file already existed.
    */
  def downloadFile(remoteUrl: String, localPath: String): Task[Boolean] =
    DataUtils.loadFile(remoteUrl, localPath)

  /** Extracts a tar.gz archive. Delegates to common.DataUtilities.extractTarGz.
    * Returns None (skipped) if either path is null, Some(()) otherwise.
    */
  def extractTarGz(inputPath: String, outputPath: String): ZIO[Any, IOException, Option[Unit]] =
    DataUtils.extractTarGz(inputPath, outputPath)
      .mapError(e => new IOException(e.getMessage, e))
      .when(inputPath != null && outputPath != null)
