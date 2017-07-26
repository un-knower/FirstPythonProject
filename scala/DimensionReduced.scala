package aduu.stat.test.spark.ml

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkContext, SparkConf}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File

/**
  * 降维模型
  * Created by Ray on 2016/11/30.
  */
object DimensionReduced {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[6]").setAppName("DimensionReduced")
      .set("spark.driver.memory","2g")
      .set("spark.excutor.memory","2g")
    val sc = new SparkContext(conf)
    val rdd = sc.wholeTextFiles("D:/chrome_download/lfw/lfw/*/*")
    val files = rdd.map{case (fileName,context) => fileName.replace("file:/","")}
    val first = files.first()
    println(first)
    println(files.count())
    val first_pic = loadImageFromFile(first)
    println(first_pic)
    // 转换成灰度图片
    val first_pic_gray = processImage(first_pic,100,100)
    println(first_pic_gray)
    println(extractPixels(first,50,50).mkString(","))
    //把图片保存到临时文件夹，便于查看
//    ImageIO.write(first_pic_gray,"jpg",new File("D:/chrome_download/lfw/lfw/AJ_Cook/AJ_Cook_0002.jpg"))

    val pixels = files.map{f => extractPixels(f,50,50)}
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))
    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache()
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    //让所有向量减去当前列的平均值
    val scaledVectors = vectors.map(v => scaler.transform(v))
    val matrix = new RowMatrix(scaledVectors)
    val k = 10
    val pc = matrix.computePrincipalComponents(k)
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows,cols)



  }


  /**
    * 读取图片
 *
    * @param path
    * @return
    */
  def loadImageFromFile(path : String): BufferedImage ={
    ImageIO.read(new File(path))
  }

  /**
    * 彩色图片转换成灰色
 *
    * @param image
    * @param width
    * @param height
    * @return
    */
  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }

  /**
    * 获取读片数据
 *
    * @param image
    * @return
    */
  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }

  def extractPixels(path: String, width: Int, height: Int):Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }
}
