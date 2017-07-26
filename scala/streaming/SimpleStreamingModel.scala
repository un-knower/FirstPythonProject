package aduu.stat.test.spark.ml.streaming

import java.util.Random

import breeze.linalg.DenseVector
import kafka.serializer.StringDecoder
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * 简单的在线计算模型
  * Created by Ray on 2017/3/22.
  */
object SimpleStreamingModel {

  val MaxEvents = 100
  val NumFeatures = 100
  val random = new Random()

  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]","SimpleStreamingModel",Seconds(5))
    val stream = ssc.socketTextStream("localhost",9999)
    val numFeatures = 100
    val zeroVector = DenseVector.zeros[Double](numFeatures)
    val model = new StreamingLinearRegressionWithSGD()
                      .setInitialWeights(Vectors.dense(zeroVector.data))
                      .setNumIterations(1)
                      .setStepSize(0.01)

    // 创建一个标签点的流
    val labeledStream = stream.map { event =>
      val split = event.split("\t")
      val y = split(0).toDouble
      val features = split(1).split(",").map(_.toDouble)
      LabeledPoint(label = y, features = Vectors.dense(features))
    }

    // 在流上训练测试模型，并打印预测结果作为展示
    model.trainOn(labeledStream)
    model.predictOn(labeledStream.map(f => f.features)).print()

    ssc.start()
    ssc.awaitTermination()

  }

  /** 生成服从正态分布的稠密向量的函数*/
  def generateRandomArray(n: Int) = Array.tabulate(n)(_ => random.nextGaussian())

  /** 生成一个确定的随机模型权重向量 */
  val w = new DenseVector(generateRandomArray(NumFeatures))
  val intercept = random.nextGaussian() * 10

  /** 生成一些随机数据事件 */
  def generateNoisyData(n: Int) = {
    (1 to n).map { i =>
      val x = new DenseVector(generateRandomArray(NumFeatures))
      val y: Double = w.dot(x)
      val noisy = y + intercept
      (noisy, x)
    }
  }
}
