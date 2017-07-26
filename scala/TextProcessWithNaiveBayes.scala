package aduu.stat.test.spark.ml

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
  * 使用贝叶斯模型进行文本分类
  * 空格分词处理后的原始文本上应用哈希单词频率转换。我们将在这些文本上训练模型，
  * 并模仿我们对使用TF-IDF特征训练的模型所做的，评估在测试集上的表现：
  * Created by Ray on 2016/12/2.
  */
object TextProcessWithNaiveBayes {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[6]").setAppName("TextProcessWithNaiveBayes")
                  .set("spark.drive.memory","4g")
    val sc = new SparkContext(conf)

    val trainRDD = sc.wholeTextFiles("D:\\chrome_download\\20news-bydate\\20news-bydate-train\\*\\*").cache()
    val testRDD = sc.wholeTextFiles("D:\\chrome_download\\20news-bydate\\20news-bydate-test\\*\\*").cache()
    // 首先拿到所有的新闻标题(其实也就是标记数据)
    val newsgroup = trainRDD.map{case (file,text) => file.split("/").takeRight(2).head}
    val newsgroupMap = newsgroup.distinct().collect().zipWithIndex.toMap

    // 简单的按空格分词
    val rowTokens = trainRDD.map{ case (file,text) => text.split(" ")}

    val dim = math.pow(2, 18).toInt  // 2的18次方==262144
    val hashingTF = new HashingTF(dim)
    // 用hash单词频率转换
    val rowTF = rowTokens.map(doc => hashingTF.transform(doc))
    val rawTrain = newsgroup.zip(rowTF).map{case (topic,vector) => LabeledPoint(newsgroupMap(topic),vector)}
    //训练贝叶斯模型
    val model = NaiveBayes.train(rawTrain,0.1)

    //在测试集上计算模型性能
    val rowTestTF = testRDD.map{case (file,text) => (newsgroupMap(file.split("/").takeRight(2).head),hashingTF.transform(text))}
    val rawTest = rowTestTF.map{case (label,text) => LabeledPoint(label,text)}
    val rawPredictAndLabel = rawTest.map(lp => (lp.label,model.predict(lp.features)))
    val rawAccuracy = rawPredictAndLabel.filter(x => x._1 == x._2).count() / rawTest.count()
    val rawMetrics = new MulticlassMetrics(rawPredictAndLabel)
    println(rawAccuracy)
    println(rawMetrics.weightedFMeasure)

  }


}
