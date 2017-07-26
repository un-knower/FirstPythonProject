package aduu.stat.test.spark.ml.dsp

import java.text.SimpleDateFormat
import java.util.Date

import aduu.stat.util.Resource
import org.apache.spark.{SparkContext, SparkConf}
import org.codehaus.jettison.json.JSONObject

/**
  * dsp广告数据特征预处理
  * 抽取了2017-04-14,2017-04-17这两天中的竞价日志来进行模型测试数据 (不能使用直投日志来做模型训练)
  * 主要是根据平均点击率跟总的点击率很相近来抽取的,下面是sql
  * SELECT a.id,ymd,b.`put_way`,req_num,show_num,click_num,click_num/show_num AS 'djl',a.`ctm`,a.`utm` FROM dsp_stat.`stat_day` a
  * INNER JOIN dsp_advert.`ad_plan` b ON a.`ad_id` = b.`id`
  * AND b.`put_way` = 'rtb'
  * WHERE req_num > 0 AND req_num < 100000
  * AND click_num/show_num > 0.008 AND click_num/show_num < 0.010
  * ORDER BY show_num DESC,click_num DESC
  * Created by Ray on 2017/5/9.
  *
  *
  * (bid_request_ex,262012)
  * (click_jump_ex,975)
  * (click_notice_ex,3)
  * (show_notice_ex,158416)
  */
object DspMLTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DspMLTest").setMaster("local[4]")
      .set("spark.executor.memory","5g")
      .set("spark.driver.memory","5g")

    val sc = new SparkContext(conf)

    val rdd = sc.textFile("D:\\soft\\bigdata\\ml_data\\train\\2017-04-17\\*\\*")
    rdd.take(1).foreach(println)
    val paris = rdd.map(f => parseJSON(f)).cache()

    // 先把点击的日志缓存起来
    paris.filter(f => f._1.nonEmpty && f._2 != null && "click_jump" == f._1).foreach(f => {
      val bid_id = f._2.optString("bid_id","")
      Resource.redisPool.withClient(client =>{
        client.auth("5ed9969f-22f7-4845-b1ef-c38001fe2a2e:ald0Ml320")
        client.set("click_"+bid_id,"1")
      })
    })



    // 再处理展示的日志
    paris.filter(f => f._1.nonEmpty && "show_notice" == f._1).foreach(f =>{
      val bid_id = f._2.optString("bid_id","")
      Resource.redisPool.withClient(client =>{
        client.auth("5ed9969f-22f7-4845-b1ef-c38001fe2a2e:ald0Ml320")
        val result = client.get("click_"+bid_id)
        if(result.nonEmpty){
          client.set("show_"+bid_id,"1")  // 1表示展示日志被点击
        }else{
          client.set("show_"+bid_id,"0")  // 0表示展示日志未点击
        }
      })
    })

    val bid_request = paris.filter(f => f._1.nonEmpty && "bid_request" == f._1).map(f => {
      val bid_id = f._2.optString("bid_id","")
      val prop = f._2.optJSONObject("properties")
      try{
        if(prop != null){
          var target = -1
          val sno = prop.optString("sno","")
          val imp_type = prop.optString("imp_type","")
          val instl = prop.optInt("instl",0)
          val width = prop.optInt("width",0)
          val height = prop.optInt("height",0)
          val channel_type = prop.optString("channel_type","")
          val site_cat_app_cat = if(channel_type == "site") prop.optString("site_cat","") else prop.optString("app_cat","")
          val make = prop.optString("make","")
          val os = prop.optString("os","")
          val connectiontype = prop.optString("connectiontype","")
          val prov_id = prop.optInt("prov_id",0)
          val city_id = prop.optInt("city_id",0)
          val time = f._2.optString("time")
          val hour = new SimpleDateFormat("H").format(new Date(time.toLong)).toInt
          Resource.redisPool.withClient(client =>{
            client.auth("5ed9969f-22f7-4845-b1ef-c38001fe2a2e:ald0Ml320")
            val result = client.get("show_"+bid_id)
            if(result.nonEmpty){
              target = result.get.toInt
            }
          })
          if(target == -1){
            null
          }else{
            sno+","+imp_type+","+instl+","+width+","+height+","+site_cat_app_cat+","+make+","+os+","+connectiontype+","+prov_id+","+city_id+","+hour+","+target
          }
        }
      }catch {
        case e : Exception => println("bid_request.map error !!")
        null
      }
    }).filter(f => f!= null).repartition(1)

    bid_request.saveAsTextFile("hdfs://192.168.1.136:9000/dsp/train_data")
  }


  def parseJSON(line:String): (String,JSONObject) = {
    if(line.isEmpty){
      ("",null)
    }else{
      var event = ""
      var sno_type = ""
      var json : JSONObject = null
      try{
        json = new JSONObject(line)
        val properties = json.optJSONObject("properties")
        event = json.optString("event","")
        sno_type = properties.optString("sno_type","")
      }catch {
        case e : Exception => println("解析JSON发生了错误,错误原因："+ e.getMessage +",错误日志："+line.toString)
      }
      if(sno_type == "ex" && "click_jump,show_notice,bid_request".contains(event)){
        (event,json)
      }else{
        ("",json)
      }
    }

  }
}

//case class TrainData(sno:String,imp_type:String,instl:Int,width:Int,height:Int,site_cat_app_cat:String,
//                      make:String,os:String,connectiontype:String,prov_id:Int,city_id:Int,hour:Int,target:Int)
