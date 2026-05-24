package speechrecognition
package imdb

import common.DataUtilities as DataUtils
import zio.*

import java.io.File
import java.nio.file.{Files, Paths}

object ImdbDataDownloader:

  private val IMDB_COMMENTS_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

  def downloadIMDBDatabase(imdbPath: String): ZIO[Any, Throwable, Unit] = {
    val archivePath   = s"$imdbPath/aclImdb_v1.tar.gz"
    val extractedPath = s"$imdbPath/aclImdb"

    for {
      _ <- ZIO.attempt(new File(imdbPath)).flatMap { dir =>
        if (!dir.exists()) ZIO.attempt(dir.mkdirs()).unit else ZIO.unit
      }

      archiveFileExists <- ZIO.attempt(Files.exists(Paths.get(archivePath)))
      extractedExists   <- ZIO.attempt(Files.exists(Paths.get(extractedPath)))

      _ <- if (!archiveFileExists)
        DataUtils.loadFile(IMDB_COMMENTS_URL, archivePath) *>
          Console.printLine(s"Downloaded to: $archivePath") *>
          DataUtils.extractTarGz(archivePath, imdbPath)
      else if (!extractedExists)
        Console.printLine("Archive exists, extracting...") *>
          DataUtils.extractTarGz(archivePath, imdbPath)
      else
        Console.printLine("Data already exists, skipping download and extraction.")
    } yield ()
  }

