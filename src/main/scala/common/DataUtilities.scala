package common

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import zio.*

import java.io.*

object DataUtilities:

  private val bufferSize = 4096

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
