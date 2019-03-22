import java.io._

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable._
import Array._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._





object ying_cheng_task1 {

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
    var b = 20
    var r = 2
    var n = 40

    val conf = new SparkConf().setAppName("Task1").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val start = System.currentTimeMillis()

//    val file_path = "/Users/irischeng/INF553/Assignment/hw3/yelp_train.csv"
    val file_path = args(0)
    val output_path = args(1)
    val file = sc.textFile(file_path)
    var header = file.first()
//    println(header)

    var file_RDD = file.filter(line => line!= header).map(s => s.split(",")).map(s => (s(1), Set(s(0))))
//    println(file_RDD.collect()(1))
    var all_user_RDD = file_RDD.map(s=>s._2).distinct()
//    println(all_user_RDD.collect()(1))
    var all_user_list = all_user_RDD.collect()
    val len = all_user_list.length


//    for (i<- all_user_list){
//      println(i)
//    }


//     build a dict for all user
    var dict_user = Map[String, Int]()
//    println(all_user_list.length)

    for (i<- range(0, all_user_list.length)){
      var temp_key = all_user_list(i).toList(0)
//      println(temp_key)
      dict_user(temp_key) = i
    }

//    println(dict_user)

    var business_user_list_RDD = file_RDD.reduceByKey(_ ++ _)
//    var business_user_list = business_user_list_RDD.collect()
//    println(business_user_list_RDD.collect()(1))

    var dict_bID_uID = Map[String, Set[String]]()
    for (i<-business_user_list_RDD.collect()){
      dict_bID_uID(i._1) = i._2
    }

    var signature_matrix = business_user_list_RDD.map(s => (s._1, create_singnature_tuple(s._1, s._2, n, len, dict_user)))
//    println(signature_matrix.collect()(1))

    var dict_eachband_bid = Map[(Int, Int, Int), ArrayBuffer[String]]()
    for (i<- signature_matrix.collect()){
//      println(i)
      for (j<- range(0, i._2.length, r)){
//        println(j)
        var band_id = j/r
        var temp_key = (band_id, i._2(j), i._2(j+1))
        if (dict_eachband_bid.contains(temp_key)){
          dict_eachband_bid(temp_key) +=  i._1
        }
        if (!dict_eachband_bid.contains(temp_key)){
          dict_eachband_bid.put(temp_key, ArrayBuffer())
          dict_eachband_bid(temp_key) += i._1
        }
      }

      }
//    println(dict_eachband_bid)

    var result = Set[(String, String, Float)]()
    var result_key = Set[(String, String)]()
    var candidate_pair_list = List()
    for (i<- dict_eachband_bid){
//      key = i._1
      if (dict_eachband_bid(i._1).length >= 2){
        var temp = dict_eachband_bid(i._1)
        var pair_list = temp.sorted.combinations(2)
        for (each_pair <- pair_list){
//          println(each_pair)
          var temp_intersection = dict_bID_uID(each_pair(0))&(dict_bID_uID(each_pair(1)))
          var temp_union = dict_bID_uID(each_pair(0)) | (dict_bID_uID(each_pair(1)))
          var j_similarity = temp_intersection.size.toFloat/temp_union.size.toFloat
//          println(temp_intersection)
//          println(temp_intersection.size)
//          println(temp_union.size)
//          println(j_similarity)
          if (j_similarity>=0.5){
            var temp_result = (each_pair(0), each_pair(1), j_similarity)
            var temp_result_key = (each_pair(0), each_pair(1))
            result.add(temp_result)
            result_key.add(temp_result_key)
          }
        }

      }
    }

//    println(result.size)

    val writer1 = new PrintWriter(output_path)
    //    writer1.write("city, stars")
    writer1.write("business_id_1, business_id_2, similarity\n")
    for (i <- result.toList.sorted) {
//      print(i.toString().stripPrefix("(").stripSuffix(")").replace(",", ", "))
//      writer1.write(i.toString().stripSuffix(")").stripPrefix("(").replace("'", " "))
      writer1.write(i.toString().stripPrefix("(").stripSuffix(")").replace(",", ", "))
      writer1.write("\n")
    }

    writer1.close()

    var end = System.currentTimeMillis()
    println("Duration:"+(end - start)/1000.0)



    }

  }


