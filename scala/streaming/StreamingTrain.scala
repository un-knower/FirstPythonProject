package aduu.stat.test.spark.ml.streaming

import org.apache.spark.mllib.regression.StreamingLinearRegressionWithSGD

/**
  * Created by Ray on 2016/8/18.
  */
object StreamingTrain {

  def main(args: Array[String]): Unit = {
    new StreamingLinearRegressionWithSGD()
  }

}
