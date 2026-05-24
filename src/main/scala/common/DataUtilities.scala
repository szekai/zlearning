package common

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import zio.*

import java.io.*
import java.net.URL
import java.nio.file.{Files, Paths, StandardCopyOption}

object DataUtilities:

  private val bufferSize = 4096

  /** Downloads a file from a URL to a local path.
    * Returns true if downloaded, false if file already existed or inputs were null.
    */
  def loadFile(remoteUrl: String, targetPath: String): Task[Boolean] =
    if remoteUrl == null || targetPath == null then ZIO.succeed(false)
    else {
      val path = Paths.get(targetPath)
      if Files.exists(path) then ZIO.succeed(false)
      else
        ZIO.attemptBlocking {
          Files.createDirectories(path.getParent)
          val connection = new URL(remoteUrl).openConnection()
          connection.setRequestProperty("User-Agent", "Mozilla/5.0")
          val in = connection.getInputStream
          try {
            Files.copy(in, path, StandardCopyOption.REPLACE_EXISTING)
          } finally in.close()
          true
        }
    }

  /** Recursively deletes a directory or file. */
  def deleteDirectory(path: String): Task[Unit] =
    ZIO.attemptBlocking {
      val file = File(path)
      if file.exists() then deleteRecursively(file)
    }

  private def deleteRecursively(f: File): Unit = {
    if f.isDirectory then {
      val children = f.listFiles()
      if children != null then children.foreach(deleteRecursively)
    }
    f.delete()
  }

  def extractTarGz(inputPath: String, outputPathRaw: String): Task[Unit] = {
    if inputPath == null || outputPathRaw == null then ZIO.unit
    else {
      val outputPath =
        if outputPathRaw.endsWith(File.separator) then outputPathRaw
        else outputPathRaw + File.separator

      ZIO.attemptBlocking {
        val fis = new FileInputStream(inputPath)
        val bis = new BufferedInputStream(fis)
        val gzipIn = new GzipCompressorInputStream(bis)
        val tarIn = new TarArchiveInputStream(gzipIn)

        val buffer = Array.ofDim[Byte](bufferSize)

        try {
          Iterator
            .continually(Option(tarIn.getNextEntry).map(_.asInstanceOf[TarArchiveEntry]))
            .takeWhile(_.isDefined)
            .flatten
            .foreach { entry =>
              val outFile = File(outputPath + entry.getName)
              if entry.isDirectory then
                outFile.mkdirs()
              else {
                val parent = outFile.getParentFile
                if parent != null then parent.mkdirs()
                val out = new BufferedOutputStream(new FileOutputStream(outFile), bufferSize)
                try {
                  Iterator
                    .continually(tarIn.read(buffer))
                    .takeWhile(_ != -1)
                    .foreach(count => out.write(buffer, 0, count))
                } finally out.close()
              }
            }
        } finally {
          tarIn.close()
        }
      }
    }
  }
