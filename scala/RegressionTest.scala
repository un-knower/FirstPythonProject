package aduu.stat.test.spark.ml

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by Ray on 2016/10/24.
  */
object RegressionTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Classification").setMaster("local")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("C:\\Users\\Ray\\Desktop\\xiaping.txt")
    val records = rawData.map(line => line.split("\t"))
//    println(records.first()(0),records.first()(3))
//    println(records.take(3))

    //最低成交价
    val zdcjjData = records.map(r =>{
      val label = r(4).toDouble
      val features = r.slice(0, r.size - 2).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    })

    //最低成交价
    val pjcjjData = records.map(r =>{
      val label = r(5).toDouble
      val features = r.slice(0, r.size - 2).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    })

    // 依次训练模型，首先训练逻辑回归模型 LinearRegressionWithSGD
    val lrModel = LinearRegressionWithSGD.train(zdcjjData,10)
    //开始预测，使用逻辑回归模型进行预测
    val dataPoint = zdcjjData.first
    val lrPrediction = lrModel.predict(dataPoint.features)
    println("逻辑回归模型第一条数据预测值："+lrPrediction+"   ...   第一条数真实值："+dataPoint.label)
  }
}
