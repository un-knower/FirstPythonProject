package aduu.stat.test.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel}
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

/**
  * Created by Ray on 2016/7/7.
  */
object MovieRecommendEvaluate {

  /**
    * 推荐模型效果的评估---均方差
    */
  def recommendAssessment(model : MatrixFactorizationModel,ratings:RDD[Rating]): Unit ={
    //predict函数也支持传入一个RDD类型参数
    /*val user_product = sc.parallelize(Array((789,123),(788,124)))
    val ss = model.predict(user_product)

    //预测用户789对于123电影的评价,predict函数可以以(user, item)ID对类型的RDD对象为输入，这时它将为每一对都
    //生成相应的预测得分
    val predictedRating = model.predict(196, 242)
    println(predictedRating)

    //推荐模型效果的评估，1.均方差，2.K值平均准确率
    //对用户已经评价的电影，根据预测数据看看是否准确
    val actualRating = ratings.take(1)(0)
    println(actualRating)
    println(math.pow(predictedRating - actualRating.rating,2))*/

    //计算出所有用户对所有电影的预测得分
    val userProducts = ratings.map{case Rating(user,product,rating) => (user,product)}
    val predicts = model.predict(userProducts).map{case Rating(user,product,rating) => ((user,product),rating)}

    //实际评价数据与测试评价数据
    val ratingAndProducts = ratings.map{case Rating(user,product,rating) => ((user,product),rating)}.join(predicts)
    val MSE = ratingAndProducts.map{
      case ((user,product),(actal,rating)) => math.pow(actal - rating,2)
    }.reduce(_ + _) / ratingAndProducts.count()

    println("均方差是："+MSE)

  }


  /**
    * 推荐模型效果的评估---K值平均准确率
    * K值平均准确率（MAPK）的意思是整个数据集上的K值平均准确率（Average Precision at K
    * metric， APK）的均值。 APK是信息检索中常用的一个指标。它用于衡量针对某个查询所返回的 “前
    * K个”文档的平均相关性。对于每次查询，我们会将结果中的前K个与实际相关的文档进行比较
    * 该值得分为0，表明该模型在相关性电影的预测上并不理想
    */
  def kValueAvgAccurate(ratings : RDD[Rating],model : MatrixFactorizationModel,sc : SparkContext): Unit ={
    //下面来计算对用户789推荐的APK指标怎么样。首先提取出用户实际评级过的电影的ID：
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    val actualMovies = moviesForUser.map(_.product)
    println("用户实际评级过的电影数据："+actualMovies.size)
    println(actualMovies.mkString("\t"))
    //    moviesForUser.sortBy(-_.rating).take(10).map(rating => (rating.product, rating.rating)).foreach(println)
    val topKRecs = model.recommendProducts(789,10)
    val predictedMovies = topKRecs.map(_.product)
    println("用户推荐的前10个电影数据：")
    println(predictedMovies.mkString("\t"))
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    //求789用户推荐结果的平均准确率
    println("789用户推荐结果的K值平均准确率 : "+apk10)


    //  全局MAPK的求解要计算对每一个用户的APK得分，再求其平均。这就要为每一个用户都生
    //  成相应的推荐列表。针对大规模数据处理时，这并不容易，但我们可以通过Spark将该计算分布
    //  式进行。不过，这就会有一个限制，即每个工作节点都要有完整的物品因子矩阵。这样它们才能
    //  独立地计算某个物品向量与其他所有物品向量之间的相关性。然而当物品数量众多时，单个节点
    //  的内存可能保存不下这个矩阵。此时，这个限制也就成了问题。
    val itemFactors = model.productFeatures.map{case (id,factor) => factor}.collect()
    val itemMatris = new DoubleMatrix(itemFactors)
    println(itemMatris.rows,itemMatris.columns)

    //在spark分布式环境上 每个工作节点都需要有一个物品因子矩阵
    val imBroadcast = sc.broadcast(itemMatris)

    val allRecs = model.userFeatures.map{ case (userId,array) =>
      val userVector = new DoubleMatrix(array)
      val scores = imBroadcast.value.mmul(userVector)
      val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId,recommendedIds)
    }

    val userMovies = ratings.map{ case Rating(user, product, rating) =>
      (user, product)}.groupBy(_._1)

    val K = 10
    val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted,
    actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K = " + MAPK)

  }

  //K值平均准确率  预测用户的前K个物品与实际的相关性的评估值
  def avgPrecisionK(actual : Seq[Int], predicted : Seq[Int], k : Int):Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p,i) <- predK.zipWithIndex){
      if(actual.contains(p)){
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if(actual.isEmpty){
      1.0
    }else{
      score / scala.math.min(actual.size,k).toDouble
    }
  }


  /**
    * 使用spark的内置函数计算
    * @param model
    * @param ratings
    */
  def sparkBuildInFunctionCalMSE(model : MatrixFactorizationModel, ratings:RDD[Rating]): Unit ={

    //计算出所有用户对所有电影的预测得分
    val userProducts = ratings.map{case Rating(user,product,rating) => (user,product)}
    val predicts = model.predict(userProducts).map{case Rating(user,product,rating) => ((user,product),rating)}

    //实际评价数据与测试评价数据
    val ratingsAndPredictions = ratings.map{case Rating(user,product,rating) => ((user,product),rating)}.join(predicts)

//    println(ratingsAndPredictions.take(10).mkString("\n"))
    val predictAndTrue = ratingsAndPredictions.map{case ((user,product),(actual,predicted)) => (predicted,actual)}
    val regressionMetrics = new RegressionMetrics(predictAndTrue)


    //直接衡量“用户与物品”评级矩阵的重建误差(用户的实际评级数据-模型预测的评级数据的平方误差)
    println("Mean Squared Error(均方差-MSE) = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error(均方根误差-SMSE) = " + regressionMetrics.rootMeanSquaredError)


  }


}
