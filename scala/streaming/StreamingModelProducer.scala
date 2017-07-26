package aduu.stat.test.spark.ml.streaming

import java.util.Random

import org.apache.spark.mllib.linalg.DenseVector

/**
  * Created by Ray on 2016/8/18.
  */
object StreamingModelProducer {

  import breeze.linalg._

  def main(args: Array[String]) {

    /*val maxEvents = 100
    val numFeatures = 100
    val random = new Random()

    /** 生成服从正态分布的稠密向量的函数*/
    def generateRandomArray(n : Int) = Array.tabulate(n)(_ => random.nextGaussian())

    println(generateRandomArray(5).mkString("\n"))

    // 生成一个确定的随机模型权重向量
    val w = new DenseVector(generateRandomArray(numFeatures))
    val intercept = random.nextGaussian() * 10

    def generateNoisyData(n : Int) ={
      (1 to n).map{i =>
        val x = new DenseVector(generateRandomArray(numFeatures))
        val y : Double = w.dot(x)
        val noisy = y + intercept
        (noisy,x)
      }
    }*/

  }



}
