package speechrecognition

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

  def extractTarGz(inputPath: String, outputPath: String): ZIO[Any, IOException, Option[Unit]] =
    common.DataUtilities.extractTarGz(inputPath, outputPath)
      .mapError(e => new IOException(e.getMessage, e))
      .when(inputPath != null && outputPath != null)
