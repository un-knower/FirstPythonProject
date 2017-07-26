package aduu.stat.test.spark.ml

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{IDF, HashingTF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
  * TF-IDF计算的标准定义如下：
  *     tf-idf(t,d) = tf(t,d) * idf(t)
  * 这里， tf(t,d)是单词t在文档d中的频率（出现的次数），
  * idf(t)是文集中单词t的逆向文本频率；定义如下：
  *    idf(t) = log( N / d )
  * 这里N是文档的总数， d是出现过单词t的文档数量
  * Created by Ray on 2016/9/6.
  */
object TextProcess {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[4]").setAppName("TextProcess")
      .set("spark.driver.memory","4g")
    val sc = new SparkContext(conf)

    val path = "D:\\chrome_download\\20news-bydate\\20news-bydate-train\\*\\*"
    val rdd = sc.wholeTextFiles(path)
    val text1 = rdd.map{
      case (file,text) => text
    }
//    println(rdd.first())
    println(text1.count())


    //打印所有的新闻主题
    val newsgroups = rdd.map{case (file,text) =>
      file.split("/").takeRight(2).head
    }
    val countByGroup = newsgroups.map(n => (n,1)).reduceByKey(_+_)
      .collect().sortBy(-_._2).mkString("\n")
    println(countByGroup)

    //文本处理先分词，先简单的按空格分词，把单词都变小写
    //按空格分词会导致很多符号都成了单词，所以不能这样分词，再按正则分词
    val text = rdd.map{case (file,text) => text}
    val whiteSpaceSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase()))
    println(whiteSpaceSplit.distinct().count())

    //再过滤掉有数字的单词
    val regex = """[^0-9]*""".r
    val filterNums = whiteSpaceSplit.filter(token => {
      regex.pattern.matcher(token).matches()
    })
    println("过滤之后的单词数量："+filterNums.distinct().count())


    //输出高频单词
    val tokenCounts = filterNums.map{t => (t,1)}.reduceByKey(_ + _)
    val orderingDesc = Ordering.by[(String,Int),Int](_._2)
    println(tokenCounts.top(20)(orderingDesc).mkString("\n"))

    //过滤掉停用词
    val stopWords = Set("the","a","an","of","or","in","for","by","on","but", "is", "not",
                        "with", "as", "was", "if",
                        "they", "are", "this", "and", "it", "have", "from", "at", "my",
                        "be", "that", "to")             //停用词列表

    val tokenCountsFilterStopWords = tokenCounts.filter{case (k,v) => !stopWords.contains(k)}
    println(tokenCountsFilterStopWords.top(20)(orderingDesc).mkString("\n"))


    //过滤掉只有一个字符的单词，这部分单词一般不会包含太多的信息
    //在整个文集中只出现一次的单词是没有价值的，因为这些单词没有足够的训练数据，把这些数据也过滤掉
    val tokenCountsFilteredSize = tokenCountsFilterStopWords.filter{case (k,v) => k.size >= 2 && v >=2}
//    println(tokenCountsFilteredSize.top(20)(orderingDesc).mkString("\n"))
    println(tokenCountsFilteredSize.distinct.count())


    val oreringAsc = Ordering.by[(String,Int),Int](-_._2)
    val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map {
      case (k, v) => k }.collect.toSet
    val tokenCountsFilteredAll = tokenCountsFilteredSize.filter { case
      (k, v) => !rareTokens.contains(k) }
//    println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))
    println("最后的特征维度："+tokenCountsFilteredAll.distinct.count())


    // 把前面的所有的过滤步骤组合到下面这个函数中
    def tokenize(line:String) : Seq[String] = {
      line.split("""\W+""").map(_.toLowerCase())
      .filter(token => regex.pattern.matcher(token).matches())
      .filterNot(token => stopWords.contains(token))
      .filterNot(token => rareTokens.contains(token))
      .filter(token => token.size > 2).toSeq
    }



//    println("  aa  "+text.flatMap(t => tokenize(t)).distinct().count())
    val tokens = text.map(doc => tokenize(doc))
    println(tokens.first.take(20))

    val dim = math.pow(2, 18).toInt  // 2的18次方==262144
    val hashingTF = new HashingTF(dim)
    // HashingTF的transform函数把每个输入文档（即词项的序列）映射到一个MLlib的Vector对象
    val tf = hashingTF.transform(tokens)
    tf.cache

    //SparseVector 这个类在两个地方要用到,所以在这里给它一个别名
    import org.apache.spark.mllib.linalg.{ SparseVector => SV }
    val v = tf.first.asInstanceOf[SV]
    println(v.size)
    println(v.values.size)
    println(v.values.take(10).toSeq)
    println(v.indices.take(10).toSeq)

    //现在通过创建新的IDF实例并调用RDD中的fit方法，利用词频向量作为输入来对文库中的
    //每个单词计算逆向文本频率。之后使用IDF的transform方法转换词频向量为TF-IDF向量：
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val v2 = tfidf.first().asInstanceOf[SV]
    println(v2.size)
    println(v2.values.size)
    println(v2.values.take(10).toSeq)
    println(v2.indices.take(10).toSeq)


    // 计算一下整个文档的TF_IDF最小和最大权值
    val minMaxVals = tfidf.map(v =>{
      val sv = v.asInstanceOf[SV]
      (sv.values.min,sv.values.max)
    })

    val globalMinMax = minMaxVals.reduce{case ((min1,max1),(min2,max2)) =>{
      (math.min(min1,min2),math.max(max1,max2))
    }}
    println(globalMinMax)


    // 我们预估两个从曲棍球新闻组随机选择的新闻比较相似
    val hockeyText = rdd.filter{ case (file,text) => file.contains("hockey")}
    val hockeyTF = hockeyText.mapValues(doc => {
      hashingTF.transform(tokenize(doc))
    })
    val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))

    // 有了曲棍球向量之后，随机取两个曲棍球的文档，并计算它们的余弦相似度(表示两个文档的相似度)
    import breeze.linalg._
    val hockey1 = hockeyTfIdf.sample(true,0.1,42).first().asInstanceOf[SV]
    val breeze1 = new SparseVector(hockey1.indices,hockey1.values,hockey1.size)

    val hockey2 = hockeyTfIdf.sample(true,0.1,43).first().asInstanceOf[SV]
    val breeze2 = new SparseVector(hockey2.indices,hockey2.values,hockey2.size)
    // 计算余弦相似度
//    val cosineSim = breeze1.dot(breeze2) / (norm(breeze1) * norm(breeze2))


    //下面把TD-IDF作为多分类的输入数据
    // 把所有的新闻主题放到一个Map里面
    val newsgroupsMap = newsgroups.distinct().collect().zipWithIndex.toMap

//    注意zip算子假设每一个RDD有相同数量的分片，并且每个对应分片有相同
//    数量的记录。如果不是这样将会失败。这里我们可以这么假设，是因为事实上
//    tfidf RDD和newsgroup RDD都是我们对相同的RDD做了一系列的map操作后
//    得到的，都保留了分片结构。
    val zipped = newsgroups.zip(tfidf)
    val train = zipped.map{case (news_topic,vector) => {
      LabeledPoint(newsgroupsMap(news_topic),vector)
    }}
    train.cache()

    // 使用贝叶斯模型训练(它可以支持多分类)
    val model = NaiveBayes.train(train,lambda = 0.1)

    // 再使用测试集来验证模型性能
    val testPath = "D:\\chrome_download\\20news-bydate\\20news-bydate-test\\*\\*"
    val testRDD = sc.wholeTextFiles(testPath)
    val testLabels = testRDD.map{case (file,text) =>
      val topic = file.split("/").takeRight(2).head
      newsgroupsMap(topic)
    }
    val testTF = testRDD.map{case (file,text) =>
      hashingTF.transform(tokenize(text))
    }
    val testTfIdf = idf.transform(testTF)
    val zippedTest = testLabels.zip(testTfIdf)
    val test = zippedTest.map{case (topic,vector) => LabeledPoint(topic,vector)}

    //使用贝叶斯模型预测测试集上的数据
    val predictionAndLabel = test.map(x => (x.label,model.predict(x.features)))
    // 计算预测的准确率和正确率
    val acuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    val metrics = new MulticlassMetrics(predictionAndLabel)
    println("准确率："+acuracy)
    println("召回率："+metrics.weightedFMeasure)



    //训练贝叶斯模型
    /*val model = NaiveBayes.train(rawTrain,0.1)

    //在测试集上计算模型性能
    val rowTestTF = testRDD.map{case (file,text) => (newsgroupMap(file.split("/").takeRight(2).head),hashingTF.transform(text))}
    val rawTest = rowTestTF.map{case (label,text) => LabeledPoint(label,text)}
    val rawPredictAndLabel = rawTest.map(lp => (lp.label,model.predict(lp.features)))
    val rawAccuracy = rawPredictAndLabel.filter(x => x._1 == x._2).count() / rawTest.count()
    val rawMetrics = new MulticlassMetrics(rawPredictAndLabel)
    println(rawAccuracy)
    println(rawMetrics.weightedFMeasure)*/






  }

}
