package aduu.stat.test.spark.ml

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * spark回归模型
  *  介绍MLlib中的各种回归模型；
    * 讨论回归模型的特征提取和目标变量的变换；
    * 使用MLlib训练回归模型；
    * 介绍如何用训练好的模型做预测；
    * 使用交叉验证研究设置不同的参数对性能的影响。
  * Created by Ray on 2016/7/20.
  *
  * Spark的MLlib库提供了两大回归模型：线性模型和决策树模型。
  * 线性回归模型本质上和对应的线性分类模型一样，唯一的区别是线性回归使用的损失函数、
  * 相关连接函数和决策函数不同。 MLlib提供了标准的最小二乘回归模型（其他广义线性回归模型
  * 也正在计划当中）。
  *
  *   instant：记录ID
    * dteday：时间
    * season：四季节信息，如spring、 summer、 winter和fall
    * yr：年份（2011或者2012）
    * mnth：月份
    * hr：当天时刻
    * holiday：是否是节假日
    * weekday：周几
    * workingday：当天是否是工作日
    * weathersit：表示天气类型的参数
    * temp：气温
    * atemp：体感温度
    * hum：湿度
    * windspeed：风速
    * casual：临时用户数
    * registered：注册用户数
    * cnt：目标变量，每小时的自行车租用量
  */
object regression {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("regression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val raw_data = sc.textFile("D:\\迅雷下载\\Bike-Sharing-Dataset\\hour_noheader.csv")
    val num_data = raw_data.count
    val records = raw_data.map(_.split(","))
    records.cache()
    val first = records.first

    println(num_data)
    println(first.mkString(","))

    val mappings = new ArrayBuffer[scala.collection.Map[String, Long]]()
    for(i <- 2 to 9){
      mappings.append(get_mapping(records,i))
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
      for(x <- 2 to 9){
        val m = mappings(i)    // Map(2 -> 1, 1 -> 3, 4 -> 0, 3 -> 2)
        val idx = m(record(x)) //
        cat_vec(idx.toInt + step) = 1
        i = i+1
        step = step + m.size
      }

      val num_vec = new Array[Double](4)
      num_vec(0) = record(10).toDouble
      num_vec(1) = record(11).toDouble
      num_vec(2) = record(12).toDouble
      num_vec(3) = record(13).toDouble
      cat_vec ++ num_vec
    }

    def extract_label(record : Array[String])={
      record(record.length-1).toDouble
    }

    val data = records.map(arr => LabeledPoint(extract_label(arr),Vectors.dense(extract_features(arr))))
    val first_point = data.first()

    println("第一条数据原始值："+first.mkString(","))
    println("第一条数据特征目标值："+first_point.label)
    println("第一条数据特征："+first_point.features)
    println("第一条数据特征长度："+first_point.features.size)


    // 为决策树创建特征向量,因为决策树模型可以直接使用原始数据(不需要将类型数据用二元向量表示)
    def extract_features_dt(record : Array[String]): Array[Double] ={
      record.slice(2,14).map(x => x.toDouble)
    }
    val data_dt = records.map(x => LabeledPoint(extract_label(x),Vectors.dense(extract_features_dt(x))))
    val first_point_dt = data_dt.first()

    println("决策树模型的第一条数据特征："+first_point_dt.features)
    println("决策树模型的第一条数据特征长度："+first_point_dt.features.size)

//    xlModel(data,data_dt)
//    yhModel(data,data_dt)
    yhModel2(data,data_dt)
  }

  def get_mapping(rdd : RDD[Array[String]], idx : Int) : scala.collection.Map[String, Long] = {
    rdd.map(arr => arr(idx)).distinct().zipWithIndex().collectAsMap()
  }


  def xlModel(data :RDD[LabeledPoint] , data_dt: RDD[LabeledPoint]){
    // 训练线性回归模型
    val linear_model = LinearRegressionWithSGD.train(data,numIterations = 10,stepSize = 0.1)
    val true_vs_predicted = data.map(x => (x.label,linear_model.predict(x.features)))
    println("线性回归模型真实值与预测值对比："+true_vs_predicted.take(5).mkString(","))

    // 训练决策树模型
    val map = Map[Int,Int]()
    val dt_model = DecisionTree.trainRegressor(data_dt,map,"variance",5,32)
    val true_vs_predicted_dt = data_dt.map(x => (x.label,dt_model.predict(x.features)))
    println("决策树回归模型真实值与预测值对比："+true_vs_predicted_dt.take(5).mkString(","))

    pgModel(true_vs_predicted,true_vs_predicted_dt)
  }



  //评估模型性能
  def pgModel(true_vs_predicted : RDD[(Double,Double)],
              true_vs_predicted_dt : RDD[(Double,Double)]): Unit ={
    val mse = true_vs_predicted.map{x => squared_error(x._1,x._2)}.mean()
    val mae = true_vs_predicted.map{x => abs_error(x._1,x._2)}.mean()
    val rmsle = Math.sqrt(true_vs_predicted.map{x => squared_log_error(x._1,x._2)}.mean())
    println("线性回归模型的mse："+mse)
    println("线性回归模型的mae："+mae)
    println("线性回归模型的rmsle："+rmsle)

    val mse_dt = true_vs_predicted_dt.map{x => squared_error(x._1,x._2)}.mean()
    val mae_dt = true_vs_predicted_dt.map{x => abs_error(x._1,x._2)}.mean()
    val rmsle_dt = Math.sqrt(true_vs_predicted_dt.map{x => squared_log_error(x._1,x._2)}.mean())
    println("决策树回归模型的mse："+mse_dt)
    println("决策树回归模型的mae："+mae_dt)
    println("决策树回归模型的rmsle："+rmsle_dt)

  }


  //优化模型
  def yhModel(data :RDD[LabeledPoint] , data_dt: RDD[LabeledPoint]): Unit ={
    //首先针对线性回归模型的目标变量取对数
    val data_log = data.map(lp => LabeledPoint(Math.log(lp.label),lp.features))
    val model_log = LinearRegressionWithSGD.train(data_log, 10, 0.1)
    // 这里因为前面进行了对数计算，现在进行指数计算转换回来
    val true_vs_predicted_log = data_log.map(x => (Math.exp(x.label),Math.exp(model_log.predict(x.features))))

    val mse = true_vs_predicted_log.map{x => squared_error(x._1,x._2)}.mean()
    val mae = true_vs_predicted_log.map{x => abs_error(x._1,x._2)}.mean()
    val rmsle = Math.sqrt(true_vs_predicted_log.map{x => squared_log_error(x._1,x._2)}.mean())
    println("优化后的线性回归模型的mse："+mse)
    println("优化后的线性回归模型的mae："+mae)
    println("优化后的线性回归模型的rmsle："+rmsle)

    //再针对决策树回归模型的目标变量取对数
    val data_log_dt = data_dt.map(lp => LabeledPoint(Math.log(lp.label),lp.features))
    val model_log_dt = DecisionTree.trainRegressor(data_log_dt,Map[Int,Int](),"variance",5,32)
    // 这里因为前面进行了对数计算，现在进行指数计算转换回来
    val true_vs_predicted_log_dt = data_log_dt.map(x => (Math.exp(x.label),Math.exp(model_log_dt.predict(x.features))))

    val mse_dt = true_vs_predicted_log_dt.map{x => squared_error(x._1,x._2)}.mean()
    val mae_dt = true_vs_predicted_log_dt.map{x => abs_error(x._1,x._2)}.mean()
    val rmsle_dt = Math.sqrt(true_vs_predicted_log_dt.map{x => squared_log_error(x._1,x._2)}.mean())
    println("优化后的决策树回归模型的mse："+mse_dt)
    println("优化后的决策树回归模型的mae："+mae_dt)
    println("优化后的决策树回归模型的rmsle："+rmsle_dt)


  }

  // 模型参数调优
  def yhModel2(data :RDD[LabeledPoint] , data_dt: RDD[LabeledPoint]): Unit ={
    // 1.创建训练集和测试集来评估参数
    val test_data = data.sample(false,0.2,42)
    val train_data = data.subtract(test_data)
    println("训练集数据大小："+test_data.count()+"\n测试集数据大小："+train_data.count()+"\n总数据大小："+data.count())
    val test_data_dt = data_dt.sample(false,0.2,42)
    val train_data_dt = data_dt.subtract(test_data_dt)
    println("训练集数据大小："+test_data_dt.count()+"\n测试集数据大小："+train_data_dt.count())

    // 测试逻辑回归模型迭代次数，步长对性能的影响，一般来讲：较小的步长与较大的迭代次数下通常可以收敛得到较好的解
    /*for(i <- Seq(1,5,10,20,50,100)){
      for(j <- Seq(0.01, 0.025, 0.05, 0.1, 1.0)) {
        val rmsle = evaluate(train_data, test_data, i, j)
        println("逻辑回归模型迭代次数，步长对性能的影响：", (i,j, rmsle))
      }
    }*/

    // 测试决策树模型迭代次数对性能的影响
    /*for(i <- Seq(1,2,3,4,5,10,20)) {
      val rmsle = evaluate_dt(train_data_dt, test_data_dt, i,32)
      println("决策树模型迭代次数对性能的影响：", (i, rmsle))
    }*/

    // 决策树模型划分数对性能的影响
    for(i <- Seq(2, 4, 8, 16, 32, 64, 100)) {
      val rmsle = evaluate_dt(train_data_dt, test_data_dt, 5,i)
      println("决策树模型划分数对性能的影响：", (i, rmsle))
    }

  }


  //计算不同参数下的RMSLE指标
  def evaluate(train :RDD[LabeledPoint],test :RDD[LabeledPoint],iterations : Int, step : Double) : Double = {
    val model = LinearRegressionWithSGD.train(train,numIterations = iterations,stepSize = step)
    val tp = test.map(p => (p.label,model.predict(p.features)))
    val rmsle = Math.sqrt(tp.map(f => squared_log_error(f._1,f._2)).mean())
    rmsle
  }

  def evaluate_dt(train :RDD[LabeledPoint],test :RDD[LabeledPoint],maxDepth:Int,huafenshu:Int) : Double = {
    val model = DecisionTree.trainRegressor(train,Map[Int,Int](),"variance",maxDepth,huafenshu)
    val tp = test.map(p => (p.label,model.predict(p.features)))
    val rmsle = Math.sqrt(tp.map(f => squared_log_error(f._1,f._2)).mean())
    rmsle
  }

  //预测值与实际值的差的平方
  def squared_error(actual : Double, pred : Double): Double = {
    Math.pow((pred - actual),2)
  }

  //预测值与实际值的差的绝对值
  def abs_error(actual : Double, pred : Double): Double = {
    Math.abs(pred - actual)
  }

  def squared_log_error(actual : Double, pred : Double): Double ={
    Math.pow((Math.log(actual + 1) - Math.log(pred + 1)),2)
  }




}
