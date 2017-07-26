package aduu.stat.test.spark.ml.Titanic

import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by Ray on 2017/6/30.
  * 对Titanic上的人员是否获救进行预测
  * #* PassengerId => 乘客ID
  * #* Pclass => 乘客等级(1/2/3等舱位)
  * #* Name => 乘客姓名
  * #* Sex => 性别
  * #* Age => 年龄
  * #* SibSp => 堂兄弟/妹个数
  * #* Parch => 父母与小孩个数
  * #* Ticket => 船票信息
  * #* Fare => 票价
  * #* Cabin => 客舱
  * #* Embarked => 登船港口
  */
object Titanic {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[2]").setAppName("Titanic")
    val sc = new SparkContext(conf)

    val train = sc.textFile("D:\\soft\\python_tools\\kaggle\\Titanic\\train_noheader.csv")
    val test = sc.textFile("D:\\soft\\python_tools\\kaggle\\Titanic\\test_noheader.csv")

//    train.take(1).foreach(println)
//    println("---" * 10)
//    test.take(1).foreach(println)


    // 最终的数据格式
    // Survived	Pclass	Sex	Age	Fare	Embarked	Title	IsAlone	Age*Class

    // 首先删除Ticket，Cabin两个字段，这两个字段看上去对预测没有什么用处
    // passengerId,Survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked
    val train_people = train.map(f => f.split(",")).map{ f => {
      if(f.length != 12) {
        println("格式不对1 : "+f(0))
        null
      }else{
        TitanicPeople(f(0),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),f(11))
      }
    }}.cache()

    val test_people = test.map(f => f.split(",")).map{ f => {
      if(f.length != 11) {
        println("格式不对2 : "+f(0))
        null
      }else{
        TitanicPeople(f(0),"",f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10))
      }
    }}.cache()

    // Mrs 表示已婚妇女，miss 表示未婚妇女，ms 表示婚姻状况未知,名称这个字段这里先暂时不管，后面再说

    //计算训练数据集的平均年龄，年龄字段有缺失值，用平均值填充
    val train_age_data = train_people.filter(f => f != null && f.age.nonEmpty).map(f => f.age.toDouble)
    val train_avg_age = Math.floor(train_age_data.sum() / train_age_data.count())

    val test_age_data = test_people.filter(f => f != null && f.age.nonEmpty).map(f => f.age.toDouble)
    val test_avg_age = Math.floor(test_age_data.sum() / test_age_data.count())
    println("训练数据的平均年龄是："+train_avg_age)
    println("测试数据的平均年龄是："+test_avg_age)


  }

}


case class TitanicPeople(passengerId: String, survived: String,pclass :String, name:String, sex: String, age:String,
                         sibsp:String, parch:String, ticket:String, fare:String, cabin:String, embarked:String)

case class TrainData(survived:Int,pclass:Int,sex:Int,age:Int,fare:Int,embarked:Int,title:Int,is_alone:Int,age_class:Int)

case class TestData(passenger_id:Int,pclass:Int,sex:Int,age:Int,fare:Int,embarked:Int,title:Int,is_alone:Int,age_class:Int)
