package aduu.stat.test.spark.ml

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Impurity, Entropy}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.{SquaredL2Updater, Updater, SimpleUpdater}

/**
  * spark 分类模型
  *
  * spark常见的三种分类模型：
  * 1.线性模型：
  *   1.1 逻辑回归
  *   1.2 线性支持向量机
  * 2.朴素贝叶斯模型(1-of-k编码的类型特征更符合朴素贝叶斯模型)
  *
  * 二分类模型的损失函数：
  * 1.逻辑损失 等价于 逻辑回归模型
  * 2.合页损失 等价于 线性支持向量机(SVM)
  *
  * 如何改进模型性能以及参数调优
  * 1.特征标准化(满足正态分布)
  * 2.使用其它特征
  * 3.使用正确的数据格式
  * 4.模型参数
  *  4.1 迭代次数
  *  4.2 步长
  *  4.3 正则化
  *
  *  模型评估：
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
  *
  * Created by Ray on 2016/6/1.
  */
object Classification {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Classification").setMaster("local")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("D:\\迅雷下载\\train.tsv")
    val records = rawData.map(line => line.split("\t"))
    println(records.first()(0))
    println(records.first()(3))

    // 由于数据格式的问题，我们做一些数据清理的工作，在处理过程中把额外的（"）去掉。数
    // 据集中还有一些用"?"代替的缺失数据，本例中，我们直接用0替换那些缺失数据：
    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    data.cache()

    val count = data.count()
    println(count)


    // 在对数据集做进一步处理之前，我们发现数值数据中包含负的特征值。我们知道，朴素贝叶
    // 斯模型要求特征值非负，否则碰到负的特征值程序会抛出错误。因此，需要为朴素贝叶斯模型构
    // 建一份输入特征向量的数据，将负特征值设为0：
    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
        "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    val (lrModel,svmModel,nbModel,dtModel) = xlModel(data,nbData)

    /**
      * 模型训练完毕之后可以将模型保存在磁盘上，下次直接读取磁盘使用该模型
      */
    lrModel.save(sc, "C:\\Users\\Ray\\Desktop\\modelSave")
    LogisticRegressionModel.load(sc,"C:\\Users\\Ray\\Desktop\\modelSave")

    pingguModel(data,nbData,lrModel,svmModel,nbModel,dtModel,count)

    standardization(data,nbData)
    addOtherCharacteristic(records,count)
    useCurrentDataFormat(records,count)
    useOtherParams(records,count)
    jueceshu(data)



  }


  /**
    * 训练模型，并简单进行预测
    *
    * @param data
    * @param nbData
    * @return
    */
  def xlModel(data : RDD[LabeledPoint],
              nbData:RDD[LabeledPoint]):(LogisticRegressionModel,SVMModel,NaiveBayesModel,DecisionTreeModel)= {
    val numIterations = 10
    val maxTreeDepth = 5

    // 依次训练模型，首先训练逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(data,numIterations)

    //训练SVM模型
    val svmModel = SVMWithSGD.train(data,numIterations)

    //训练朴素贝叶斯模型,使用处理过的没有负值的数据集
    val nbModel = NaiveBayes.train(nbData,numIterations)

    //训练决策树模型,在决策树中，我们设置模式或者Algo时使用了Entropy不纯度估计。
    val dtModel = DecisionTree.train(data,Algo.Classification,Entropy,maxTreeDepth)

    //开始预测，使用逻辑回归模型进行预测
    val dataPoint = data.first
    val lrPrediction = lrModel.predict(dataPoint.features)
    println("逻辑回归模型第一条数据预测值："+lrPrediction+"   ...   第一条数真实值："+dataPoint.label)

    val svmPrediction = svmModel.predict(dataPoint.features)
    println("线性支持向量机模型第一条数据预测值："+svmPrediction+"   ...   第一条数真实值："+dataPoint.label)

    val nbDataPoint = nbData.first
    val nbPrediction = nbModel.predict(nbDataPoint.features)
    println("朴素贝叶斯模型第一条数据预测值："+nbPrediction+"   ...   第一条数真实值："+nbDataPoint.label)

    val dtPrediction = dtModel.predict(dataPoint.features)
    println("决策树模型第一条数据预测值："+dtPrediction+"   ...   第一条数真实值："+dataPoint.label)
    /*val predictions = lrModel.predict(data.map(lp => lp.features))
    println("辑回归模型前5条数据预测值："+predictions.take(5).mkString(","))
    data.take(5).foreach(x => println("辑回归模型前5条数据真实值："+x.label))*/

    (lrModel,svmModel,nbModel,dtModel)
  }

  /**
    * 评估模型性能
    *
    * @param data
    * @param nbData
    * @param lrModel
    * @param svmModel
    * @param nbModel
    * @param dtModel
    * @param count
    */
  def pingguModel(
       data : RDD[LabeledPoint],
       nbData : RDD[LabeledPoint],
       lrModel : LogisticRegressionModel,
       svmModel : SVMModel,
       nbModel : NaiveBayesModel,
       dtModel : DecisionTreeModel,
       count : Long): Unit ={

    //评估逻辑回归模型性能
    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    println("逻辑回归正确率:"+ lrTotalCorrect / count)


    //评估SVM模型性能
    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    println("SVM模型正确率:"+ svmTotalCorrect /count)


    //评估朴素贝叶斯模型性能
    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    println("朴素贝叶斯模型正确率:"+ nbTotalCorrect / nbData.count)


    //评估决策树模型性能
    val dtTotalCorrect = data.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum
    println("决策树模型正确率:"+ dtTotalCorrect / count)
  }


  /**
    * 特征标准化
    */
  def standardization(
       data:RDD[LabeledPoint],
       nbData:RDD[LabeledPoint]): Unit ={

      val vectors = data.map(_.features)
      val matrix = new RowMatrix(vectors)
      val matrixSummary = matrix.computeColumnSummaryStatistics()

      println("计算矩阵每列的均值:"+matrixSummary.mean) //计算矩阵每列的均值
      println("计算矩阵每列的最小值:"+matrixSummary.min)  //计算矩阵每列的最小值
      println("计算矩阵每列的最大值:"+matrixSummary.max)  //计算矩阵每列的最大值
      println("计算矩阵每列的方差:"+matrixSummary.variance)  //计算矩阵每列的方差
      println("矩阵每列中非0项的数目:"+matrixSummary.numNonzeros)  //矩阵每列中非0项的数目

      val scaler = new StandardScaler(withMean = true,withStd = true).fit(vectors)
      val scaledData = data.map(lp => LabeledPoint(lp.label,scaler.transform(lp.features)))
      println("标准化前的特征向量："+data.first.features)
      println("标准化后的特征向量："+scaledData.first.features)


      //使用标准化后的数据进行逻辑回归模型训练
      val lrModelScaler = LogisticRegressionWithSGD.train(scaledData,10)
      val lrTotalCorrentScaled = scaledData.map{ point =>
        if(lrModelScaler.predict(point.features) == point.label) 1 else 0
      }.sum
      val lrAccuracyScaled = lrTotalCorrentScaled / data.count  //标准化之后的逻辑回归模型的正确率
      val lrPredictionsVsTrue  = scaledData.map{point =>
      (lrModelScaler.predict(point.features),point.label)
      }
      val lrModelScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)

      val lrPr = lrModelScaled.areaUnderPR
      val lrRoc = lrModelScaled.areaUnderROC

      //    println("特征标准化之后的逻辑回归模型的正确率："+(lrAccuracyScaled * 100 ) +"% ," +(lrPr * 100 )+"% , "+(lrRoc * 100 )+"%")
      println(
      f"${lrModelScaled.getClass.getSimpleName}" +"\n"+
      f"Accuracy(特征标准化之后的逻辑回归模型的正确率):${lrAccuracyScaled * 100}%2.4f%%" +"\n"+
      f"Area under PR: ${lrPr * 100.0}%2.4f%%" +"\n"+
      f"Area under ROC: ${lrRoc * 100.0}%2.4f%%")


      val svmModelScaler = SVMWithSGD.train(scaledData,10)
      val svmTotalCorrentScaled = scaledData.map{ point =>
      if(svmModelScaler.predict(point.features) == point.label) 1 else 0
      }.sum
      val svmAccuracyScaled = svmTotalCorrentScaled / data.count  //标准化之后的SVM(线性支持向量机)模型的正确率
      println("特征标准化之后的线性支持向量机模型的正确率"+svmAccuracyScaled)


      //决策树、朴素贝叶斯模型不受数据特征化的影响
      /*val dtModelScaler = DecisionTree.train(scaledData,Algo.Classification,Entropy,5)
      val dtTotalCorrentScaled = scaledData.map{ point =>
      if(dtModelScaler.predict(point.features) == point.label) 1 else 0
      }.sum
      val dtAccuracyScaled = dtTotalCorrentScaled / data.count  //标准化之后的决策树模型的正确率
      println("特征标准化之后的决策树模型的正确率"+dtAccuracyScaled)*/


}


  /**
    * 加入类别特征
    */
  def addOtherCharacteristic(records : RDD[Array[String]],count : Long): RDD[LabeledPoint] ={
    val categories = records.map(r => r(3)).distinct().collect().zipWithIndex.toMap
    val numCategories = categories.size
    println(categories)
    println(numCategories)

    val dataCategories = records.map{ r =>
      val trimmed = r.map(_.replace("\"",""))
      val label = trimmed(r.size - 1).toInt
      val categoriesIdx = categories(r(3))
      val categoriesFeature = Array.ofDim[Double](numCategories)
      categoriesFeature(categoriesIdx) = 1.0

      val otherFeature = trimmed.slice(4,r.size - 1).map(d => if(d == "?") 0.0 else d.toDouble)
      val features = categoriesFeature ++ otherFeature
      LabeledPoint(label,Vectors.dense(features))
    }
    println(dataCategories.first())

    //对数据进行标准化转换
    val scalerCats = new StandardScaler(withMean = true,withStd = true)
      .fit(dataCategories.map(lp => lp.features))

    val scaledDataCats = dataCategories.map{lp =>
      LabeledPoint(lp.label,scalerCats.transform(lp.features))
    }

    //加入类别特征，标准化之前的数据
    println("加入类别特征，标准化之前的数据："+dataCategories.first.features)
    //加入类别特征，标准化之后的数据
    println("加入类别特征，标准化之后的数据："+scaledDataCats.first.features)

    //训练逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(scaledDataCats,10)
    val lrTotalCurrectScaledCats = scaledDataCats.map{point =>
      if(point.label == lrModel.predict(point.features)) 1 else 0
    }.sum

    val lrAccuracyScaledCats = lrTotalCurrectScaledCats/count
    println("加入类别特征，逻辑回归模型的正确率："+lrAccuracyScaledCats)

    scaledDataCats
  }


  /**
    * 为朴素贝叶斯模型建立正确的输入格式
    * (只使用类别特征转换的向量)
    *
    * @param records
    * @param count
    */
  def useCurrentDataFormat(records : RDD[Array[String]] , count : Long): Unit ={
    val categories = records.map(r => r(3)).distinct().collect().zipWithIndex.toMap
    val numCategories = categories.size

    val nbData = records.map{r =>
      val trimmed = r.map(_.replace("\"",""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      LabeledPoint(label,Vectors.dense(categoryFeatures))
    }

    val nbModelCats = NaiveBayes.train(nbData)

    val nbAccuracyCats = nbData.map{point =>
      if(point.label == nbModelCats.predict(point.features)) 1 else 0
    }.sum / count

    println("只使用类别特征转换的向量作为特征,朴素贝叶斯模型的正确率为："+nbAccuracyCats)

  }


  /**
    * 参数调优
    */
  def useOtherParams(records : RDD[Array[String]],count : Long) : Unit ={
    val categories = records.map(r => r(3)).distinct().collect().zipWithIndex.toMap
    val numCategories = categories.size
    println(categories)
    println(numCategories)

    val dataCategories = records.map{ r =>
      val trimmed = r.map(_.replace("\"",""))
      val label = trimmed(r.size - 1).toInt
      val categoriesIdx = categories(r(3))
      val categoriesFeature = Array.ofDim[Double](numCategories)
      categoriesFeature(categoriesIdx) = 1.0

      val otherFeature = trimmed.slice(4,r.size - 1).map(d => if(d == "?") 0.0 else d.toDouble)
      val features = categoriesFeature ++ otherFeature
      LabeledPoint(label,Vectors.dense(features))
    }
    println(dataCategories.first())

    //对数据进行标准化转换
    val scalerCats = new StandardScaler(withMean = true,withStd = true)
      .fit(dataCategories.map(lp => lp.features))

    val scaledDataCats = dataCategories.map{lp =>
      LabeledPoint(lp.label,scalerCats.transform(lp.features))
    }


    //迭代次数对结果的影响
    val iterResults = Seq(1,5,10,50).map{param =>
      val model = trainWithParams(scaledDataCats,0.0,param,new SimpleUpdater,1.0)
      createMetrics(s"$param iterations", scaledDataCats, model)
    }
    iterResults.foreach(println)


    //步长对结果的影响
    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, 10, new
          SimpleUpdater, param)
      createMetrics(s"$param step size", scaledDataCats, model)
    }
    stepResults.foreach(println)

    //正则化对结果的影响
    val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, 10,
        new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter",
        scaledDataCats, model)
    }
    regResults.foreach(println)

  }

  /**
    * 参数调优，辅助函数，根据给定输入训练模型
    */
  def trainWithParams(input : RDD[LabeledPoint],regParam : Double,numIterations : Int,
                      updater : Updater,stepSize : Double) ={
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations)
      .setUpdater(updater)
      .setRegParam(regParam)
      .setStepSize(stepSize)
    lr.run(input)
  }



  def createMetrics(label : String, data : RDD[LabeledPoint],model : ClassificationModel) = {
    val scoreAndLabels = data.map{
      point => (model.predict(point.features),point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label,metrics.areaUnderROC())
  }



  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int,
                        impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  /**
    * 决策树模型调优(不需要对数据进行标准化)
    * 决策树训练深度越深，准确率越高
    *
    */
  def jueceshu(data : RDD[LabeledPoint]): Unit ={
    //使用Entropy不纯度并改变树的训练深度
    val dtResultsEntropy = Seq(1,2,3,4,5,10,20).map{param =>
      val model = trainDTWithParams(data,param,Entropy)
      val scoreAndLabels = data.map{point =>
        val score = model.predict(point.features)
        (if(score > 0.5) 1.0 else 0.0 , point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree ",metrics.areaUnderROC())
    }

    dtResultsEntropy.foreach{println}
  }


}
