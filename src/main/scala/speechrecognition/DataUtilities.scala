package speechrecognition

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClientBuilder
import zio.*

import java.io.*
import java.nio.file.{Files, Paths}

object DataUtilities:

  def downloadFile(remoteUrl: String, localPath: String): Task[Boolean] =
    if remoteUrl == null || localPath == null then ZIO.succeed(false)
    else {
      val path = Paths.get(localPath)
      if Files.exists(path) then ZIO.succeed(false)
      else
        for {
          _ <- ZIO.attempt(Files.createDirectories(path.getParent))
          downloaded <- ZIO.scoped {
            ZIO.acquireRelease(
              ZIO.attempt(HttpClientBuilder.create().build())
            )(client => ZIO.attempt(client.close()).orDie).flatMap { client =>
              ZIO.attempt(client.execute(HttpGet(remoteUrl))).flatMap { response =>
                ZIO.acquireRelease(ZIO.succeed(response))(r => ZIO.attempt(r.close()).orDie).flatMap { res =>
                  Option(res.getEntity) match
                    case Some(entity) =>
                      ZIO.attemptBlocking {
                        val out = Files.newOutputStream(path)
                        try entity.writeTo(out)
                        finally out.close()
                      }.as(true)
                    case None => ZIO.succeed(false)
                }
              }
            }
          }
          exists <- ZIO.attempt(Files.exists(path))
          _ <- ZIO.fail(new IOException(s"File doesn't exist: $localPath")).when(!exists)
        } yield downloaded
    }

  def extractTarGz(inputPath: String, outputPath: String): ZIO[Any, IOException, Option[Unit]] = {
    val outputDir = if outputPath.endsWith(File.separator) then outputPath else outputPath + File.separator
    val bufferSize = 4096

    ZIO.scoped {
      ZIO.attemptBlockingIO {
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
              val outFile = File(outputDir + entry.getName)
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
  }.unless(inputPath == null || outputPath == null)
