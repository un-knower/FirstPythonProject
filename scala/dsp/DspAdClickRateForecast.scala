package aduu.stat.test.spark.ml.dsp

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable.ArrayBuffer

/**
  * dsp广告点击率的预测
  * Created by Ray on 2017/5/12.
  * *  模型评估：
  *  1.正确率等于训练样本中被正确分类的数目除以总样本数。类似地，错误率等于训练样本中被错误分类的样本数目除以总样本数。
  *  2.在二分类问题中，准确率定义为真阳性的数目除以真阳性和假阳性的总数，其中真阳性是指被正确预测的类别为1的样本，
  *    假阳性是错误预测为类别1的样本。如果每个被分类器预测为类别1的样本确实属于类别1，那准确率达到100%。
  *  3.召回率定义为真阳性的数目除以真阳性和假阴性的和，其中假阴性是类别为1却被预测为0
  * 的样本。如果任何一个类型为1的样本没有被错误预测为类别0（即没有假阴性） ，那召回率达到100%
  * 通常，准确率和召回率是负相关的，高准确率常常对应低召回率，反之亦然。
  *
  * 真实情况| 预测结果(正例) | 预测结果(负例)
     正例		    TP(真正例)		   FN(假反例)
     负例       FP(假正例)      TN(真反例)
  */
object DspAdClickRateForecast {

  // 正负样本数量极度不平衡，首先需要解决这个问题，
  // 并且在能取到的特征中，都不是最相关的特征，基于用户的个人信息，网页的信息都没有
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DspMLTest").setMaster("local[4]")
      .set("spark.executor.memory","5g")
      .set("spark.driver.memory","5g")

    val sc = new SparkContext(conf)
    val records = sc.textFile("D:\\soft\\bigdata\\ml_data\\train\\2017-04-17特征排列.txt").map(f => f.split(",")).filter(f => f.length > 8)


    val mappings = new ArrayBuffer[scala.collection.Map[String, Long]]()
    Array(0,1,6,7).foreach(f => {
      mappings.append(get_mapping(records,f))
    })


    def get_mapping(rdd : RDD[Array[String]], idx : Int) : scala.collection.Map[String, Long] = {
      rdd.map(arr => {
        try{
          arr(idx)
        }catch{
          case e:Exception => println("数组越界是那一条记录：" + arr.mkString(","),arr.length)
          ""
        }
      }).distinct().zipWithIndex().collectAsMap()
    }

    var cat_len = 0
    for(map <- mappings){
      cat_len += map.size
    }

    println(mappings)
    println(cat_len)


    def extract_features(record : Array[String]) : Array[Double] = {
      val cat_vec = new Array[Double](cat_len)  // 初始化数组，默认元素全是0
      var i = 0
      var step = 0

      Array(0,1,6,7).foreach(x => {
        val m = mappings(i)    // Map(2 -> 1, 1 -> 3, 4 -> 0, 3 -> 2)
        val idx = m(record(x)) //
        cat_vec(idx.toInt + step) = 1
        i = i+1
        step = step + m.size
      })

      val num_vec = new Array[Double](5)
      num_vec(0) = record(3).toDouble
      num_vec(1) = record(4).toDouble
      num_vec(2) = if(record(9) == "" || record(9) == "unknown") 0 else record(9).toDouble
      num_vec(3) = record(10).toDouble
      num_vec(4) = record(11).toDouble
      cat_vec ++ num_vec
    }

    def extract_label(record : Array[String])={
      record(record.length-1).toDouble
    }

    val data = records.map(arr => LabeledPoint(extract_label(arr),Vectors.dense(extract_features(arr))))
    val lrModel = LogisticRegressionWithSGD.train(data,10)

    //开始预测，使用逻辑回归模型进行预测
//    val dataPoint = data.first

    val count = data.count()   // 总样本大小
    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum

    // 真阳性数目
    val zhenyangCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label && point.label == 1) 1 else 0
    }.sum

    // 假阳性数目
    val jiayangCorrect = data.map { point =>
      if (lrModel.predict(point.features) == 1 && point.label == 0) 1 else 0
    }.sum

    println("逻辑回归正确率:"+ lrTotalCorrect,count,lrTotalCorrect / count)


    println("逻辑回归准确率:"+ zhenyangCorrect,jiayangCorrect,zhenyangCorrect / (zhenyangCorrect + jiayangCorrect))
  }

}
