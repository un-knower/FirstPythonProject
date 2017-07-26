package aduu.stat.test.spark.ml

import breeze.linalg._
import breeze.numerics.pow
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating, ALS}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * spark聚类模型
  * Created by Ray on 2016/8/31.
  */
object Clustering {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("Clustering")
    val sc = new SparkContext(conf)

    //电影详细数据
    val movies = sc.textFile("D:\\迅雷下载\\ml-100k\\u.item")
    println(movies.first)

    //电影题材标签
    val genre = sc.textFile("D:\\迅雷下载\\ml-100k\\u.genre")
    genre.take(5).foreach(println)

   val genrenMap = genre.filter(!_.isEmpty).map(line => {
      val arr = line.split("\\|")
      (arr(1),arr(0))
    }).collectAsMap()
    println(genrenMap)

    val titlesAndGenre = movies.map(_.split("\\|")).map(array =>{
      val genres = array.toSeq.slice(5,array.size) //截取后面的数据
      val genresAssigned = genres.zipWithIndex.filter{case (g,ids) => {
          g == "1"
      }}.map{case (g, idx) => {
          genrenMap(idx.toString)
        }}
      (array(0).toInt,(array(1),genresAssigned))
    })
    println(titlesAndGenre.first())


    //训练推荐模型，把推荐模型返回的数据作为聚类模型的输入
    val rowData = sc.textFile("D:\\迅雷下载\\ml-100k\\u.data")
    val rowRatings = rowData.map(_.split("\t").take(3))
    val ratings = rowRatings.map{ case Array(user,movie,rating) => Rating(user.toInt,movie.toInt,rating.toDouble)}.cache()
    val alsModel = ALS.train(ratings, 50, 10, 0.1)

    //电影聚类分析
    movieTrain(alsModel,titlesAndGenre)
  }


  /**
    * 电影聚类分析
    * @param alsModel
    * @param titlesAndGenre
    */
  def movieTrain(alsModel : MatrixFactorizationModel,titlesAndGenre : RDD[(Int,(String,Seq[String]))]): Unit ={
    //对推荐模型数据做处理.以便能作为聚类模型的输入
    val movieFactors = alsModel.productFeatures.map{case (id,factor) => (id,Vectors.dense(factor))}
    val movieVectors = movieFactors.map(_._2)

    val userFactors = alsModel.userFeatures.map{case (id,factor) => (id, Vectors.dense(factor))}
    val userVectors = userFactors.map(_._2)
    //    alsModel.productFeatures.take(1).foreach(a => println(a._1+" , ",a._2.mkString(",")))
    //    alsModel.userFeatures.take(1).foreach(a => println(a._1+" , ",a._2.mkString(",")))

    //在训练聚类模型之前，观察数据的特征向量分布，这可以告诉我们是否需要对训练数据进行归一化
    val movieMatrix = new RowMatrix(movieVectors)
    val moveiMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()

    println("Movei factor mean: " + moveiMatrixSummary.mean)
    println("Movei factor variance: " + moveiMatrixSummary.variance)
    println("User factor mean: " + userMatrixSummary.mean)
    println("User factor variance: " + userMatrixSummary.variance)


    val numClusters = 5    //模型参数
    val numIterations = 10 //最大迭代次数
    val numRuns = 3        //训练次数
    val movieClusterModel = KMeans.train(movieVectors,numClusters,numIterations,numRuns)
//    val userClusterModel = KMeans.train(userVectors,numClusters,numIterations,numRuns)

    val movie1 = movieVectors.first()
    val movieCluster =  movieClusterModel.predict(movie1)
    println("聚类模型预测单个电影类型："+movieCluster)

    val predictions = movieClusterModel.predict(movieVectors)
    println("聚类模型预测多个电影类型："+predictions.take(10).mkString(","))

    //上面打印的类簇ID没有内在含义，都是从0开始生成，这部分需要人工解释
    def computeDistance(v1:DenseVector[Double],v2:DenseVector[Double]) = pow(v1 -v2 , 2).sum

    //我们计算每个电影特征向量到所属类簇中心向量的距离，让结果具有可读性，输出结果添加了电影标题跟题材数据
    val titleWithFactors = titlesAndGenre.join(movieFactors)

    val moviesAssigned = titleWithFactors.map{
      case (id,((title,genre),vector)) => {
        val pred = movieClusterModel.predict(vector)
        val clusterCenter = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(clusterCenter.toArray),DenseVector(vector.toArray))
        (id,title,genre.mkString(" "), pred, dist)
      }
    }
    val clusterAssignments = moviesAssigned.groupBy { case (id,title,
    genre, cluster, dist) => cluster }.collectAsMap()

    for((k,v) <- clusterAssignments.toSeq.sortBy(_._1)){
      println(s"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map{ case (_,title,genre,_,d) =>
        (title,genre,d)
      }.mkString("\n"))
      println("==================\n")
    }
  }




}
