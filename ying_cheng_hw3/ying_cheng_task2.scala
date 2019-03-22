import java.io._

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable._
import Array._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

import scala.io.Source
import scala.math._





object ying_cheng_task2 {

  def create_singnature_tuple(b_id:String, user_id_set: Set[String], n:Int, len: Int, dict_user:Map[String, Int]): List[Int] = {
    val a_list=List(271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,929,937,941,947,953,967)

    var signature_list = List[Int]()
    for (i<- range(1,n+1)){
      var a = a_list(i)
      var b = a_list(i+1)
      var m = len

      var temp_index_list = List[Int]()
      for (user_id <- user_id_set) {
        var temp_index = (a * dict_user(user_id) + b) % m
        temp_index_list = temp_index +: temp_index_list
      }
      signature_list = (temp_index_list).min +: signature_list
    }
    return signature_list
  }

  def main()(args: Array[String]) {
    val conf = new SparkConf().setAppName("Task2").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

      val case_number = args(2).toInt
      val train_filepath = args(0)
      val val_filepath = args(1)
      val output_filepath = args(3)

//    val case_number = 1
//    val train_filepath = "/Users/irischeng/INF553/Assignment/hw3/yelp_train.csv"
//    val val_filepath = "/Users/irischeng/INF553/Assignment/hw3/yelp_val.csv"
//    val output_filepath = "case1.csv"

    if (case_number == 1) {

      val start = System.currentTimeMillis()
      var train_data = sc.textFile(train_filepath)
      var train_header = train_data.first()
      var train_RDD = train_data.filter(line => line != train_header).map(s => s.split(",")).map(s => (s(0), s(1), s(2)))
      var train_user = train_RDD.map(s => s._1).distinct().collect().toSet
      var train_business = train_RDD.map(s => s._2).distinct().collect().toSet

      //
      var val_data = sc.textFile(val_filepath)
      var val_header = val_data.first()
      var val_RDD = val_data.filter(line => line != val_header).map(s => s.split(",")).map(s => (s(0), s(1), s(2)))
      var val_user = val_RDD.map(s => s._1).distinct().collect().toSet
      var val_business = val_RDD.map(s => s._2).distinct().collect().toSet
      //          var val_ratings = val_RDD.map(s=>)


      var all_user = (train_user | val_user).toList
      var all_business = (train_business | val_business).toList

      var dict_all_user = Map[String, Int]()
      for (i <- range(0, all_user.length)) {
        var temp_key = all_user(i).toString
        dict_all_user(temp_key) = i
      }
      //      println(dict_all_user)

      var dict_all_business = Map[String, Int]()
      for (i <- range(0, all_business.length)) {
        var temp_key = all_business(i).toString
        dict_all_business(temp_key) = i
      }

      var train_ratings = train_RDD.map(s => Rating(dict_all_user(s._1), dict_all_business(s._2), s._3.toFloat))
      //      println(train_ratings.take(1))
      var val_data_ratings = val_RDD.map(s => Rating(dict_all_user(s._1), dict_all_business(s._2), s._3.toFloat))

      val rank = 3
      val numIterations = 8
      val model = ALS.train(train_ratings, rank, numIterations, 0.01)


      // Evaluate the model on rating data
      val usersProducts = val_data_ratings.map { case Rating(user, product, rate) =>
        (user, product)
      }

      val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

      val ratesAndPreds = val_data_ratings.map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.leftOuterJoin(predictions)

      var temp_res = ratesAndPreds.collect()

      val writer1 = new PrintWriter(output_filepath)
      var MSE = 0.0
      var count = 0.0
      //    writer1.write("city, stars")
      writer1.write("user_id, business_id, prediction\n")
      for (i <- temp_res) {
        //        println(i)
        if (i._2._2.isDefined) {
//          if ((i._2._2, 5).mi._2._2<=5)
          if (i._2._2.get < 5 | i._2._2.get ==5 ){
            MSE += (i._2._1 - i._2._2.get) * (i._2._1 - i._2._2.get)
            count = count + 1
          }
          if(i._2._2.get >5){
            MSE += (i._2._1 - 3.8) * (i._2._1 - 3.8)
            count = count + 1
          }

        }
        //      var temp_str = dict_all_user()
        writer1.write(all_user(i._1._1) + "," + all_business(i._1._2) + "," + i._2._2.getOrElse(3.8) + "\n")
      }


      writer1.close()

      println("Root Mean Squared Error = " + math.sqrt(MSE / count))
      var end = System.currentTimeMillis()
      println("Duration:" + (end - start) / 1000.0)


    }

    if (case_number == 2) {
      var start = System.currentTimeMillis()

      var train = Source.fromFile(train_filepath, "UTF-8").getLines()

      var dict_bID_uIDlist = Map[String, Set[String]]()
      //      # 't_xXPh62lfkPc_wgF-Iasg': ['n5kvsYIIPUVrARu0pkaazQ', 'F_0Rf6KGokgemEBep0v3Tg', 'pN6pzJR6mK7549M0azoaxg']

      //      var dict_uID_sumrating_count = Map[String, (Float,Float)]()
      //      #  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]
      var dict_uID_sum = Map[String, Float]()

      var dict_uId_count = Map[String, Float]()

      var dict_uID_ave = Map[String, Float]()
      //      # 'DlXeHPut0_ZYBJXNRFjuEA': 4.238095238095238,

      var dict_uID_bIDlist = Map[String, Set[String]]()
      //      # 'k0QpEqseu2y87KIHyGVlzw': ['9tfw-OEfpF0qC2hSzRks6g', 'o8MxZKmLhdrRk8EoRX8Sog']

      var dict_uIDandbID_rate = Map[(String, String), Float]()

      for (line <- train) {
        if (!line.contains("user_id, business_id, stars")) {
          var temp_line = line.replace("\n", "").split(",").toList
          //          println(temp_line)

          if (dict_bID_uIDlist.contains(temp_line(1))) {
            dict_bID_uIDlist(temp_line(1)).add(temp_line(0))
          }
          if (!dict_bID_uIDlist.contains(temp_line(1))) {
            dict_bID_uIDlist(temp_line(1)) = Set()
            dict_bID_uIDlist(temp_line(1)).add(temp_line(0))
          }

          if (dict_uID_bIDlist.contains(temp_line(0))) {
            dict_uID_bIDlist(temp_line(0)).add(temp_line(1))
          }
          if (!dict_uID_bIDlist.contains(temp_line(0))) {
            dict_uID_bIDlist(temp_line(0)) = Set()
            dict_uID_bIDlist(temp_line(0)).add(temp_line(1))
          }

          dict_uIDandbID_rate((temp_line(0), temp_line(1))) = temp_line(2).toFloat

          if (dict_uID_sum.contains(temp_line(0))) {
            dict_uID_sum(temp_line(0)) = dict_uID_sum(temp_line(0)) + temp_line(2).toFloat
          }
          if (!dict_uID_sum.contains(temp_line(0))) {
            dict_uID_sum(temp_line(0)) = 0
            dict_uID_sum(temp_line(0)) = dict_uID_sum(temp_line(0)) + temp_line(2).toFloat
          }

          if (dict_uId_count.contains(temp_line(0))) {
            dict_uId_count(temp_line(0)) = dict_uId_count(temp_line(0)) + 1
          }
          if (!dict_uId_count.contains(temp_line(0))) {
            dict_uId_count(temp_line(0)) = 0
            dict_uId_count(temp_line(0)) = dict_uID_sum(temp_line(0)) + 1
          }

        }
      }
      //
      for (i <- dict_uID_sum) {
        var temp_uID = i._1
        var temp_ave = i._2 / dict_uId_count(temp_uID)
        dict_uID_ave(temp_uID) = temp_ave
      }

      //      println(dict_uID_ave)

      val writer1 = new PrintWriter(output_filepath)
      writer1.write("user_id, business_id, prediction\n")

      var test = Source.fromFile(val_filepath, "UTF-8").getLines()
      var sum_difference: Float = 0
      var n = 0
      for (line <- test) {
        if (!line.contains("user_id, business_id, stars")) {
          n = n + 1
          var temp_line = line.replace("\n", "").split(",").toList
          var active_user = temp_line(0)
          var active_item = temp_line(1)
          //          println(active_user)
          //          println(active_item)

          var prediction: Float = 3.5.toFloat
          if (!dict_bID_uIDlist.contains(active_item)) {
            prediction = dict_uID_ave(active_user)
            //            println("1")

          }
          if (dict_bID_uIDlist.contains(active_item)) {
            //            println("2")
            if (!dict_uID_bIDlist.contains(active_user)) {
              prediction = 3.5.toFloat
              //              println("3")
            }
            if (dict_uID_bIDlist.contains(active_user)) {
              //              println("4")

              var similar_user_list = dict_bID_uIDlist(active_item)
              var prediction_number_sum: Float = 0
              var prediction_divide_sum: Float = 0
              for (user <- similar_user_list) {
                //                println(similar_user_list)
                if (user != active_user) {
                  var temp_weight: Float = 0
                  var corated_item = dict_uID_bIDlist(user) & dict_uID_bIDlist(active_user)
                  //                  println(corated_item)
                  var weight_number_sum: Float = 0
                  var weight_divide_active_sum: Float = 0
                  var weight_divide_user_sum: Float = 0
                  if (corated_item.size == 0) {
                    temp_weight = 0
                  }
                  if (corated_item.size != 0) {
                    for (each_item <- corated_item) {
                      //                      println(each_item)
                      var temp_active_user = dict_uIDandbID_rate((active_user, each_item)) - dict_uID_ave(active_user)
                      var temp_user = dict_uIDandbID_rate((user, each_item)) - dict_uID_ave(user)
                      //                      var temp = temp_active_user*temp_user
                      weight_number_sum = weight_number_sum + temp_active_user * temp_user
                      weight_divide_active_sum = weight_divide_active_sum + temp_active_user * temp_active_user
                      weight_divide_user_sum = weight_divide_user_sum + temp_user * temp_user
                    }
                    if ((pow(weight_divide_active_sum, 0.5) * pow(weight_divide_user_sum, 0.5)) != 0) {
                      temp_weight = (weight_number_sum / (pow(weight_divide_active_sum, 0.5) * pow(weight_divide_user_sum, 0.5))).toFloat
                    }
                    if ((pow(weight_divide_active_sum, 0.5) * pow(weight_divide_user_sum, 0.5)) == 0) {
                      temp_weight = 0
                    }

                  }
                  prediction_number_sum = prediction_number_sum + temp_weight * (dict_uIDandbID_rate((user, active_item)) - dict_uID_ave(user))
                  prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                }
              }
              if (prediction_divide_sum != 0) {
                prediction = dict_uID_ave(active_user) + prediction_number_sum / prediction_divide_sum
              }
              if (prediction_divide_sum == 0) {
                prediction = dict_uID_ave(active_user)
              }

            }
          }
          var record = (active_user, active_item, prediction)
          writer1.write(record.toString().stripPrefix("(").stripSuffix(")").replace("'", ""))
          writer1.write("\n")
          sum_difference = sum_difference + (prediction - temp_line(2).toFloat) * (prediction - temp_line(2).toFloat)
        }
      }
      var MSE = sum_difference / n
      var RMSE = pow(MSE, 0.5)
      //      println(RMSE)

      writer1.close()

      println("Root Mean Squared Error = " + RMSE)
      var end = System.currentTimeMillis()
      println("Duration:" + (end - start) / 1000.0)


    }

    if (case_number == 3) {
      var start = System.currentTimeMillis()

      var train = Source.fromFile(train_filepath, "UTF-8").getLines()

      var dict_bID_uIDlist = Map[String, Set[String]]()
      //      # 't_xXPh62lfkPc_wgF-Iasg': ['n5kvsYIIPUVrARu0pkaazQ', 'F_0Rf6KGokgemEBep0v3Tg', 'pN6pzJR6mK7549M0azoaxg']

      //      var dict_uID_sumrating_count = Map[String, (Float,Float)]()
      //      #  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]
      var dict_uID_sum = Map[String, Float]()

      var dict_uId_count = Map[String, Float]()

      var dict_uID_ave = Map[String, Float]()
      //      # 'DlXeHPut0_ZYBJXNRFjuEA': 4.238095238095238,
      var dict_bID_sum = Map[String, Float]()

      var dict_bID_count = Map[String, Float]()

      var dict_bID_ave = Map[String, Float]()

      var dict_uID_bIDlist = Map[String, Set[String]]()
      //      # 'k0QpEqseu2y87KIHyGVlzw': ['9tfw-OEfpF0qC2hSzRks6g', 'o8MxZKmLhdrRk8EoRX8Sog']

      var dict_uIDandbID_rate = Map[(String, String), Float]()

      for (line <- train) {
        if (!line.contains("user_id, business_id, stars")) {
          var temp_line = line.replace("\n", "").split(",").toList
          //          println(temp_line)

          if (dict_bID_uIDlist.contains(temp_line(1))) {
            dict_bID_uIDlist(temp_line(1)).add(temp_line(0))
          }
          if (!dict_bID_uIDlist.contains(temp_line(1))) {
            dict_bID_uIDlist(temp_line(1)) = Set()
            dict_bID_uIDlist(temp_line(1)).add(temp_line(0))
          }

          if (dict_uID_bIDlist.contains(temp_line(0))) {
            dict_uID_bIDlist(temp_line(0)).add(temp_line(1))
          }
          if (!dict_uID_bIDlist.contains(temp_line(0))) {
            dict_uID_bIDlist(temp_line(0)) = Set()
            dict_uID_bIDlist(temp_line(0)).add(temp_line(1))
          }

          dict_uIDandbID_rate((temp_line(0), temp_line(1))) = temp_line(2).toFloat

          if (dict_uID_sum.contains(temp_line(0))) {
            dict_uID_sum(temp_line(0)) = dict_uID_sum(temp_line(0)) + temp_line(2).toFloat
          }
          if (!dict_uID_sum.contains(temp_line(0))) {
            dict_uID_sum(temp_line(0)) = 0
            dict_uID_sum(temp_line(0)) = dict_uID_sum(temp_line(0)) + temp_line(2).toFloat
          }

          if (dict_uId_count.contains(temp_line(0))) {
            dict_uId_count(temp_line(0)) = dict_uId_count(temp_line(0)) + 1
          }
          if (!dict_uId_count.contains(temp_line(0))) {
            dict_uId_count(temp_line(0)) = 0
            dict_uId_count(temp_line(0)) = dict_uID_sum(temp_line(0)) + 1
          }

          if (dict_bID_sum.contains(temp_line(1))) {
            dict_bID_sum(temp_line(1)) = dict_bID_sum(temp_line(1)) + temp_line(2).toFloat
          }
          if (!dict_bID_sum.contains(temp_line(1))) {
            dict_bID_sum(temp_line(1)) = 0
            dict_bID_sum(temp_line(1)) = dict_bID_sum(temp_line(1)) + temp_line(2).toFloat
          }

          if (dict_bID_count.contains(temp_line(1))) {
            dict_bID_count(temp_line(1)) = dict_bID_count(temp_line(1)) + 1
          }
          if (!dict_bID_count.contains(temp_line(1))) {
            dict_bID_count(temp_line(1)) = 0
            dict_bID_count(temp_line(1)) = dict_bID_count(temp_line(1)) + 1
          }


        }
      }
      //
      for (i <- dict_uID_sum) {
        var temp_uID = i._1
        var temp_ave = i._2 / dict_uId_count(temp_uID)
        dict_uID_ave(temp_uID) = temp_ave
      }

      for (i <- dict_bID_sum) {
        var temp_bID = i._1
        var temp_ave = i._2 / dict_bID_count(temp_bID)
        dict_bID_ave(temp_bID) = temp_ave
      }

      //            println(dict_bID_ave)

      val writer1 = new PrintWriter(output_filepath)
      writer1.write("user_id, business_id, prediction\n")

      var test = Source.fromFile(val_filepath, "UTF-8").getLines()
      var sum_difference: Float = 0
      var n = 0
      for (line <- test) {
        if (!line.contains("user_id, business_id, stars")) {
          n = n + 1
          var temp_line = line.replace("\n", "").split(",").toList
          var active_user = temp_line(0)
          var active_item = temp_line(1)
          //                    println(active_user)
          //                    println(active_item)

          var prediction: Float = 3.5.toFloat
          if (!dict_uID_bIDlist.contains(active_user)) {
            prediction = dict_bID_ave(active_item)
            //                        println("1")

          }
          if (dict_uID_bIDlist.contains(active_user)) {
            //                        println("2")
            if (!dict_bID_uIDlist.contains(active_item)) {
              prediction = dict_uID_ave(active_user)
              //                            println("3")
            }
            if (dict_bID_uIDlist.contains(active_item)) {
              //                            println("4")

              var similar_business_list = dict_uID_bIDlist(active_user)
              var prediction_number_sum: Float = 0
              var prediction_divide_sum: Float = 0
              for (business <- similar_business_list) {
                //                println(similar_user_list)
                if (business != active_item) {
                  var temp_weight: Float = 0
                  var corated_user = dict_bID_uIDlist(business) & dict_bID_uIDlist(active_item)
                  //                  println(corated_item)
                  var weight_number_sum: Float = 0
                  var weight_divide_active_sum: Float = 0
                  var weight_divide_business_sum: Float = 0
                  if (corated_user.size == 0) {
                    temp_weight = 0
                  }
                  if (corated_user.size != 0) {
                    for (each_user <- corated_user) {
                      //                      println(each_item)
                      var temp_active_item = dict_uIDandbID_rate((each_user, active_item)) - dict_bID_ave(active_item)
                      var temp_businiess = dict_uIDandbID_rate((each_user, business)) - dict_bID_ave(business)
                      //                      var temp = temp_active_user*temp_user
                      weight_number_sum = weight_number_sum + temp_active_item * temp_businiess
                      weight_divide_active_sum = weight_divide_active_sum + temp_active_item * temp_active_item
                      weight_divide_business_sum = weight_divide_business_sum + temp_businiess * temp_businiess
                    }
                    if ((pow(weight_divide_active_sum, 0.5) * pow(weight_divide_business_sum, 0.5)) != 0) {
                      temp_weight = (weight_number_sum / (pow(weight_divide_active_sum, 0.5) * pow(weight_divide_business_sum, 0.5))).toFloat
                    }
                    if ((pow(weight_divide_active_sum, 0.5) * pow(weight_divide_business_sum, 0.5)) == 0) {
                      temp_weight = 0
                    }

                  }
                  prediction_number_sum = prediction_number_sum + temp_weight * (dict_uIDandbID_rate((active_user, business)) - dict_bID_ave(business))
                  prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                }
              }
              if (prediction_divide_sum != 0) {
                prediction = dict_bID_ave(active_item) + prediction_number_sum / prediction_divide_sum
              }
              if (prediction_divide_sum == 0) {
                prediction = dict_bID_ave(active_item)
              }

            }
          }
          var record = (active_user, active_item, prediction)
          writer1.write(record.toString().stripPrefix("(").stripSuffix(")").replace("'", ""))
          writer1.write("\n")
          sum_difference = sum_difference + (prediction - temp_line(2).toFloat) * (prediction - temp_line(2).toFloat)
        }
      }
      var MSE = sum_difference / n
      var RMSE = pow(MSE, 0.5)
      //      println(RMSE)

      writer1.close()

      println("Root Mean Squared Error = " + RMSE)
      var end = System.currentTimeMillis()
      println("Duration:" + (end - start) / 1000.0)

    }

    if (case_number == 4) {

      var b = 20
      var r = 2
      var n = 40


      val start = System.currentTimeMillis()

      //    val file_path = "/Users/irischeng/INF553/Assignment/hw3/yelp_train.csv"
      //      val file_path = args(0)
      //      val output_path = args(1)
      val file = sc.textFile(train_filepath)
      var header = file.first()
      //    println(header)

      var file_RDD = file.filter(line => line != header).map(s => s.split(",")).map(s => (s(1), Set(s(0))))
      //    println(file_RDD.collect()(1))
      var all_user_RDD = file_RDD.map(s => s._2).distinct()
      //    println(all_user_RDD.collect()(1))
      var all_user_list = all_user_RDD.collect()
      val len = all_user_list.length

      var business_rating_RDD= file.filter(line =>line!= header).map(s=>s.split(",")).map(s=>(s(1), s(2)))


      //    for (i<- all_user_list){
      //      println(i)
      //    }
      var dict_business_rating = Map[String, Float]()
      for (i<- business_rating_RDD.collect()){
        dict_business_rating(i._1) = i._2.toFloat
      }

//      println(dict_business_rating)


      //     build a dict for all user
      var dict_user = Map[String, Int]()
      //    println(all_user_list.length)

      for (i <- range(0, all_user_list.length)) {
        var temp_key = all_user_list(i).toList(0)
        //      println(temp_key)
        dict_user(temp_key) = i
      }

      //    println(dict_user)

      var business_user_list_RDD = file_RDD.reduceByKey(_ ++ _)
      //    var business_user_list = business_user_list_RDD.collect()
      //    println(business_user_list_RDD.collect()(1))

      var dict_bID_uID = Map[String, Set[String]]()
      for (i <- business_user_list_RDD.collect()) {
        dict_bID_uID(i._1) = i._2
      }

      var signature_matrix = business_user_list_RDD.map(s => (s._1, create_singnature_tuple(s._1, s._2, n, len, dict_user)))
      //    println(signature_matrix.collect()(1))

      var dict_eachband_bid = Map[(Int, Int, Int), ArrayBuffer[String]]()
      for (i <- signature_matrix.collect()) {
        //      println(i)
        for (j <- range(0, i._2.length, r)) {
          //        println(j)
          var band_id = j / r
          var temp_key = (band_id, i._2(j), i._2(j + 1))
          if (dict_eachband_bid.contains(temp_key)) {
            dict_eachband_bid(temp_key) += i._1
          }
          if (!dict_eachband_bid.contains(temp_key)) {
            dict_eachband_bid.put(temp_key, ArrayBuffer())
            dict_eachband_bid(temp_key) += i._1
          }
        }

      }
      //    println(dict_eachband_bid)

      var result = Set[(String, String, Float)]()
      var result_key = Set[(String, String)]()
      var candidate_pair_list = List()
      var dict_bID_bID_similar = Map[(String, String), Float]()
      var single_key_set = Set[String]()
      for (i <- dict_eachband_bid) {
        //      key = i._1
        if (dict_eachband_bid(i._1).length >= 2) {
          var temp = dict_eachband_bid(i._1)
          var pair_list = temp.sorted.combinations(2)
          for (each_pair <- pair_list) {
            //          println(each_pair)
            var temp_intersection = dict_bID_uID(each_pair(0)) & (dict_bID_uID(each_pair(1)))
            var temp_union = dict_bID_uID(each_pair(0)) | (dict_bID_uID(each_pair(1)))
            var j_similarity = temp_intersection.size.toFloat / temp_union.size.toFloat
            //          println(temp_intersection)
            //          println(temp_intersection.size)
            //          println(temp_union.size)
            //          println(j_similarity)
            if (j_similarity >= 0.5) {
              var temp_result = (each_pair(0), each_pair(1), j_similarity)
              var temp_result_key = (each_pair(0), each_pair(1))
              single_key_set.add(each_pair(0))
              single_key_set.add(each_pair(1))
              result.add(temp_result)
              result_key.add(temp_result_key)
              dict_bID_bID_similar(temp_result_key) = j_similarity
            }
          }

        }
      }

//      println(dict_bID_bID_similar)

      val writer1 = new PrintWriter(output_filepath)
      //    writer1.write("city, stars")
      writer1.write("user_id, business_id, prediction\n")

      var test = Source.fromFile(val_filepath, "UTF-8").getLines()
      var sum_difference: Float = 0
      var m = 0
      for (line <- test) {
        if (!line.contains("user_id, business_id, stars")) {
          var prediction: Float = 3.8.toFloat
          m = m + 1
          var temp_line = line.replace("\n", "").split(",").toList
          var active_user = temp_line(0)
          var active_business = temp_line(1)
          var prediction_number_sum: Float = 0
          var prediction_divide_sum: Float = 0
          if (single_key_set.contains(active_business)) {
            for (k <- dict_bID_bID_similar) {
              //              println(k)
              var temp_key = k._1
              var temp_weight:Float = 0
              var temp_rate:Float = 0
              if (active_business == temp_key._1) {
                temp_weight = dict_bID_bID_similar(temp_key)
                temp_rate = dict_business_rating(temp_key._2)
                prediction_number_sum = prediction_number_sum + temp_weight * temp_rate
                prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
              }
              if (active_business == temp_key._2) {
                temp_weight = dict_bID_bID_similar(temp_key)
                temp_rate = dict_business_rating(temp_key._1)
                prediction_number_sum = prediction_number_sum + temp_weight * temp_rate
                prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
              }
            }

            if (prediction_divide_sum != 0) {
              prediction = prediction_number_sum / prediction_divide_sum
            }
            if (prediction_divide_sum == 0) {
              prediction = 3.8.toFloat
            }

          }
          if (!single_key_set.contains(active_business)){
            prediction = 3.8.toFloat
          }
          var record = (active_user,active_business,prediction)
          writer1.write(record.toString().stripPrefix("(").stripSuffix(")").replace("'", ""))
          writer1.write("\n")
          sum_difference = sum_difference + (prediction - temp_line(2).toFloat) * (prediction - temp_line(2).toFloat)

        }
      }
      var MSE = sum_difference/m
      var RMSE = pow(MSE,0.5)
      writer1.close()

      val writer2 = new PrintWriter("ying_cheng_explanation.txt")
      writer2.write("Compared with the result of case3, the accuracy is reduced, but the speed is improved a lot. By using the LSH algorithm, some pairs that may be similar are found. When calculating the predicted value, the calculation amount becomes smaller and the speed becomes faster.")
      writer2.close()

      println("Root Mean Squared Error = " + RMSE)
      var end = System.currentTimeMillis()
      println("Duration:" + (end - start) / 1000.0)
        }

      }
    }
