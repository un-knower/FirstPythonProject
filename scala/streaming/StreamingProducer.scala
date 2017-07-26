package aduu.stat.test.spark.ml.streaming

import java.io.PrintWriter
import java.net.ServerSocket

import scala.util.Random

/**
  * Created by Ray on 2017/3/22.
  */
object StreamingProducer {

  def main(args: Array[String]) {
    val random = new Random()

    // 每秒最大活动数
    val maxEvents = 6

    //名称
    val names = "Miguel,Eric,James,Juan,Shawn,James,Doug,Gary,Frank,Janet,Michael,James,Malinda,Mike,Elaine,Kevin,Janet,Richard,Saul,Manuela".split(",").toSeq

    // 生成一系列可能的产品
    val products = Seq(
      "iPhone Cover" -> 9.99,
      "Headphones" -> 5.49,
      "Samsung Galaxy Cover" -> 8.95,
      "iPad Cover" -> 7.49
    )

    def generateProductEvents(n : Int) = {
      (1 to n).map{i =>
        val (product,price) = products(random.nextInt(products.size))
        val user = random.shuffle(names).head
        (user, product, price)
      }
    }

    val listener = new ServerSocket(9999)
    println("Listening on port 9999")

    while(true){
      val socket = listener.accept()
      new Thread(){
        override def run = {
          println("Got client connected from: " +
            socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(),true)
          while (true) {
            Thread.sleep(1000)
            val num = random.nextInt(maxEvents)
            val productEvents = generateProductEvents(num)
            productEvents.foreach{ event =>
              out.write(event.productIterator.mkString(","))
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
}
