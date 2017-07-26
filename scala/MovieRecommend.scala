package aduu.stat.test.spark.ml

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating, ALS}
import org.jblas.DoubleMatrix
/**
  * Created by Ray on 2016/4/21.
  * 电影推荐
  *
  * 矩阵分解(因子分解)模型：矩阵降维，分解成用户矩阵跟物品矩阵，进行用户对某物品预测时，
  * 只需要将该用户矩阵对应的行乘以该物品矩阵对应的列即可
  */
object MovieRecommend {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("JustTest")
    val sc = new SparkContext(conf)

    //获取用户信息数据
//    getUserData(sc)

    //获取电影信息数据
    getMovieData(sc)

    //获取用户评价电影信息数据
    getRatingData(sc)

  }

  /**
    * 获取用户对电影的评价数据
    *
    * @param sc
    */
  def getRatingData(sc : SparkContext){
    // u.data => user id、movie id、rating（从1到5）、timestamp
    val rating_data = sc.textFile("D:\\迅雷下载\\ml-100k\\u.data")

    //rating_fields
    val rating_fields = rating_data.map(line => line.split("\t").take(3))

    //使用电影评级数据来训练模型,处理隐式数据用trainImplicit函数 -> ALS.trainImplicit()
    val ratings = rating_fields.map{ case Array(user,movie,rating) => Rating(user.toInt,movie.toInt,rating.toDouble)}
    val model = ALS.train(ratings,50,10,0.01)


    //预测用户789对于123电影的评价,predict函数可以以(user, item)ID对类型的RDD对象为输入
    // 这时它将为每一对都生成相应的预测得分
   /* val predictedRating = model.predict(789, 123)
    println(predictedRating)

    //为用户生成K个推荐物品
    val topKRecs = model.recommendProducts(789,10)
    println(topKRecs.mkString("\n"))

    //校验推荐结果
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    println(moviesForUser.size)
    moviesForUser.sortBy(-_.rating).take(10).foreach(println)*/


    //使用余弦计算物品相似度
//    cosineCalProductSimilarity(model)

    //使用余弦计算用户相似度
//    cosineCalUserSimilarity(model)

    //模型评估-均方差评估
//    MovieRecommendEvaluate.recommendAssessment(model,ratings)
    //模型评估-K值平均准确率评估
//    MovieRecommendEvaluate.kValueAvgAccurate(ratings,model,sc)

    //使用spark内置函数来评估模型性能
    MovieRecommendEvaluate.sparkBuildInFunctionCalMSE(model,ratings)

  }


  /**
    * 使用余弦计算用户相似度
    * @param model
    * @return
    */
  def cosineCalUserSimilarity(model : MatrixFactorizationModel) : Unit = {

    val userId = 789
    val userFactor = model.userFeatures.lookup(userId).head
    val userVector = new DoubleMatrix(userFactor)
    println("789用户与自己的相似度："+cosineSimilarity(userVector,userVector))

    val sims = model.userFeatures.map{case (userId,factor) => {
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector,userVector)
      (userId,sim)
    }}

    //对相似度排序，取前10,第一种方式全局排序，取前10
    sims.sortBy(_._2,false).take(10).foreach(println)
    println("*************************************")
    //第二种方式，分区排序取前10
    sims.top(10)(Ordering.by[(Int,Double),Double]{_._2}).foreach(println)

  }

  /**
    * 使用余弦计算物品相似度
    *
    * @param model
    */
  def cosineCalProductSimilarity(model : MatrixFactorizationModel): Unit ={
    //使用自己实现的余弦相似度来衡量相似度
    //    val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    println("itemFactor : "+itemFactor)
    println("567电影与自己的相似度："+cosineSimilarity(itemVector,itemVector))

    val sims = model.productFeatures.map{ case (id,factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector,itemVector)
      (id,sim)
    }

    val sortedSims = sims.top(10)(Ordering.by[(Int,Double),Double]{
      case (id,similarity) =>  similarity
    })

    println("使用余弦相似度来计算物品之间的相似性： "+sortedSims.take(10).mkString("\n"))

  }

  def cosineSimilarity(vec1 : DoubleMatrix, vec2 : DoubleMatrix) : Double ={
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }


  /**
    * 获取电影信息数据
    *
    * @param sc
    */
  def getMovieData(sc : SparkContext){
    // u.item => movie id、title、release date(发布日期)、IMDB link
    val movie_data = sc.textFile("D:\\迅雷下载\\ml-100k\\u.item")
//    movie_data.take(1).foreach(println)

    //user_fields
//    val movie_fields = movie_data.map(line => line.replace("|",";").split(";"))
    val movie_fields = movie_data.map(line => line.split("\\|"))

    println(movie_fields.count)
    //计算电影的年龄分布(当前年份减去电影的发布年份)
    val moviesRange = movie_fields.map(f => {
      if(f(2) != null && f(2).split("-").length == 3){
        (2016-(f(2).split("-")(2).toInt),1)
      }else{
        (1900,1)
      }
    }).reduceByKey(_+_).sortByKey(false) //.sortBy(_._1)
    moviesRange.foreach(println)
  }

  /**
    * 获取用户信息数据
    *
    * @param sc
    */
  def getUserData(sc : SparkContext){
    // u.user => id、age、gender、occupation(职业)、ZIP code(邮编)
    val user_data = sc.textFile("D:\\迅雷下载\\ml-100k\\u.user")
    //    user_data.foreach(println)
    user_data.take(1).foreach(println)

    //user_fields
    val user_fields = user_data.map(line => line.replace("|",";").split(";"))

    val num_users = user_fields.map(field => field(0)).distinct().count()
    val num_genders = user_fields.map(field => field(2)).distinct().count()
    val num_occupations = user_fields.map(field => field(3)).distinct().count()
    val num_zipcodes = user_fields.map(field => field(4)).distinct().count()

    println ("Users: %d, genders: %d, occupations: %d, ZIP codes: %d".format(num_users, num_genders,
      num_occupations, num_zipcodes))

    //计算用户的年龄分布
    val agesRange = user_fields.map(f => (f(1).toInt,1)).reduceByKey(_+_).sortByKey(false) //.sortBy(_._1)
    agesRange.foreach(println)

    //计算用户职业分布,使用reduceByKey方式
    val occupationsRange = user_fields.map(f => (f(3),1)).reduceByKey(_+_).sortBy(_._2,false)
    occupationsRange.foreach(println)

    //计算用户职业分布,使用countByValue方式
    val occupationsRange1 = user_fields.map(f => (f(3))).countByValue()
    occupationsRange1.foreach(println)


  }

}
