package utils

import org.apache.commons.compress.archivers.tar.*
import zio.*

import java.io.FileOutputStream
import java.nio.file.Path
import java.util.zip.GZIPOutputStream

object TestUtils {

  def createTarGz(targetPath: Path, fileName: String, content: String): UIO[Unit] =
    ZIO.attemptBlockingIO {
      val fos = new FileOutputStream(targetPath.toFile)
      val gzos = new GZIPOutputStream(fos)
      val tarOut = new TarArchiveOutputStream(gzos)
      tarOut.setLongFileMode(TarArchiveOutputStream.LONGFILE_GNU)

      val data = content.getBytes("UTF-8")
      val entry = new TarArchiveEntry(fileName)
      entry.setSize(data.length)

      tarOut.putArchiveEntry(entry)
      tarOut.write(data)
      tarOut.closeArchiveEntry()
      tarOut.close()
    }.orDie
}
