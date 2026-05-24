package speechrecognition
package imdb

import zio.*
import java.io.{File, FileOutputStream}
import java.net.URL
import java.nio.file.{Files, Paths, StandardCopyOption}
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream

object ImdbDataDownloader:

//  private val IMDB_DATA_PATH     = "data/imdb"
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
        downloadFile(IMDB_COMMENTS_URL, archivePath) *>
          Console.printLine(s"Downloaded to: $archivePath") *>
          extractTarGz(archivePath, imdbPath)
      else if (!extractedExists)
        Console.printLine("Archive exists, extracting...") *>
          extractTarGz(archivePath, imdbPath)
      else
        Console.printLine("Data already exists, skipping download and extraction.")
    } yield ()
  }

  def downloadFile(url: String, targetPath: String): Task[Unit] = ZIO.attempt {
    val connection = new URL(url).openConnection()
    connection.setRequestProperty("User-Agent", "Mozilla/5.0")
    val in = connection.getInputStream
    try {
      Files.copy(in, Paths.get(targetPath), StandardCopyOption.REPLACE_EXISTING)
    } finally {
      in.close()
    }
  }

  private def extractTarGz(archivePath: String, targetDir: String): Task[Unit] = ZIO.attempt {
    val tarIn = new TarArchiveInputStream(
      new GzipCompressorInputStream(Files.newInputStream(Paths.get(archivePath)))
    )

    var entry = tarIn.getNextEntry
    while (entry != null) {
      if (entry.isDirectory) {
        new File(targetDir, entry.getName).mkdirs()
      } else {
        val destFile = new File(targetDir, entry.getName)
        destFile.getParentFile.mkdirs()
        val outStream = new FileOutputStream(destFile)
        tarIn.transferTo(outStream)
        outStream.close()
      }
      entry = tarIn.getNextEntry
    }
    tarIn.close()
  }

