from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext
import itertools
import os
import time
import sys
import hashlib
import math


case = int(sys.argv[3])
train_filepath = sys.argv[1]
val_filepath = sys.argv[2]
output_filepath = sys.argv[4]

if case == 1:
    # start = time.perf_counter()
    # n = 40
    # sys.argv[1]
    sc = SparkContext('local[*]', 'Task1')
    start = time.perf_counter()
    # train_filepath= "/Users/irischeng/INF553/Assignment/hw3/yelp_train.csv"
    # train_filepath = sys.argv[1]
    train_data = sc.textFile(train_filepath)
    train_header = train_data.first()
    train_RDD = train_data.filter(lambda row: row != train_header).map(lambda s: s.split(',')).map(lambda s: (s[0], s[1], s[2])).persist()
    train_ratings = train_RDD.map(lambda s: (dict_all_user[s[0]], dict_all_business[s[1]], float(s[2])))
    train_user = set(train_RDD.map(lambda s:s[0]).distinct().collect())
    train_business = set(train_RDD.map(lambda s:s[1]).distinct().collect())
    # print(type(train_user))
    # print(len(train_user))
    # print(len(train_business))
    # val_filepath = "/Users/irischeng/INF553/Assignment/hw3/yelp_val.csv"
    # val_filepath = sys.argv[2]
    val_data = sc.textFile(val_filepath)
    val_header = val_data.first()
    val_RDD = val_data.filter(lambda row: row != train_header).map(lambda s: s.split(',')).map(lambda s: (s[0], s[1], s[2])).persist()
    val_data_ratings = val_RDD.map(lambda s: (dict_all_user[s[0]], dict_all_business[s[1]], float(s[2])))
    val_user = set(val_RDD.map(lambda s:s[0]).distinct().collect())
    val_business = set(val_RDD.map(lambda s: s[1]).distinct().collect())
    # print(len(val_user))
    # print(len(val_business))

    all_user = list(train_user| val_user)
    all_business = list(train_business| val_business)
    # print(type(all_user))
    # print(len(all_user))
    # print(len(all_business))

    dict_all_user = {}
    for i in range(len(all_user)):
        temp_key = all_user[i]
        dict_all_user[temp_key]= i

    # print(dict_all_user)

    dict_all_business = {}
    for i in range(len(all_business)):
        temp_key = all_business[i]
        dict_all_business[temp_key] = i

    # print(dict_all_business)

#
    # Build the recommendation model using Alternating Least Squares
    rank = 3
    numIterations = 8
    model = ALS.train(train_ratings, rank, numIterations)

# Evaluate the model on training data
    val_data_no_rating = val_data_ratings.map(lambda s: (s[0], s[1]))  #(userid, busid)
    predictions = model.predictAll(val_data_no_rating).map(lambda r: ((r[0], r[1]), min(5,abs(r[2]))))
    # predictions_output = predictions.map(lambda r:(all_user[r[0][0]], all_business[r[0][1]], r[1])).collect()
    # print(predictions.take(1))
    ratesAndPreds = val_data_ratings.map(lambda r: ((r[0], r[1]), r[2])).leftOuterJoin(predictions)
    temp_res = ratesAndPreds.collect()

    fileObject = open(output_filepath, 'w')
    MSE = 0.0
    count = 0
    fileObject.write('user_id, business_id, prediction\n')
    for line in temp_res:
        if line[1][1] is not None:
            MSE += (line[1][0] - line[1][1]) ** 2
            count += 1
            temp_line = line[1][1]
        else:
            temp_line = 3.8
        fileObject.write(all_user[line[0][0]] + "," + all_business[line[0][1]] + "," + str(temp_line) + "\n")
    fileObject.close()
    print("Root Mean Squared Error = " + str(math.sqrt(MSE / count)))


    # print(ratesAndPreds.take(2))
    # MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x, y: x + y) / ratesAndPreds.count()
    # RMSE = math.sqrt(MSE)
    # print("Root Mean Squared Error = " + str(RMSE))

    # fileObject = open(output_filepath, 'w')
    # fileObject.write("user_id, business_id, prediction\n")
    # for i in sorted(predictions_output):
    #     fileObject.write(str(i).strip("()").replace("'", ""))
    #     fileObject.write("\n")
    # fileObject.close()

    end = time.perf_counter()
    print("duration:", end-start)

    # print(predictions_output)

if case==2:

    start = time.perf_counter()

    train = open(train_filepath, 'r')
    train_header = train.readline()

    dict_bID_uIDlist = {}
    # 't_xXPh62lfkPc_wgF-Iasg': ['n5kvsYIIPUVrARu0pkaazQ', 'F_0Rf6KGokgemEBep0v3Tg', 'pN6pzJR6mK7549M0azoaxg']

    dict_uID_sumrating_count = {}
    #  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]

    dict_uID_ave = {}
    # 'DlXeHPut0_ZYBJXNRFjuEA': 4.238095238095238,

    dict_uID_bIDlist = {}
    # 'k0QpEqseu2y87KIHyGVlzw': ['9tfw-OEfpF0qC2hSzRks6g', 'o8MxZKmLhdrRk8EoRX8Sog']

    dict_uIDandbID_rate = {}
    #  ('8tX-EprIfJarL2n88bnqkw', 'UvFRZR-i7p21YQVoB-8f1g'): 5.0

    for line in train.readlines():
        # print(line)
        if line != train_header:
            temp_line = line.replace("\n", "").split(",")
            # print(temp_line)

            if temp_line[1] in dict_bID_uIDlist:
                dict_bID_uIDlist[temp_line[1]].add(temp_line[0])
            else:
                dict_bID_uIDlist[temp_line[1]] = set()
                dict_bID_uIDlist[temp_line[1]].add(temp_line[0])
            # print(dict_bID_uIDlist)

            if temp_line[0] in dict_uID_bIDlist:
                dict_uID_bIDlist[temp_line[0]].add(temp_line[1])
            else:
                dict_uID_bIDlist[temp_line[0]] = set()
                dict_uID_bIDlist[temp_line[0]].add(temp_line[1])

            dict_uIDandbID_rate[(temp_line[0], temp_line[1])] = float(temp_line[2])

            if temp_line[0] in dict_uID_sumrating_count:
                dict_uID_sumrating_count[temp_line[0]][0] = dict_uID_sumrating_count[temp_line[0]][0] + float(
                    temp_line[2])
                dict_uID_sumrating_count[temp_line[0]][1] = dict_uID_sumrating_count[temp_line[0]][1] + 1
            else:
                dict_uID_sumrating_count[temp_line[0]] = [0, 0]
                dict_uID_sumrating_count[temp_line[0]][0] = dict_uID_sumrating_count[temp_line[0]][0] + float(
                    temp_line[2])
                dict_uID_sumrating_count[temp_line[0]][1] = dict_uID_sumrating_count[temp_line[0]][1] + 1

    for i in dict_uID_sumrating_count:
        # print(i)
        dict_uID_ave[i] = dict_uID_sumrating_count[i][0] / dict_uID_sumrating_count[i][1]

    fileObject = open(output_filepath, 'w')
    fileObject.write("user_id, business_id, prediction\n")

    test = open(val_filepath, 'r')
    test_header = test.readline()
    sum_difference = 0
    n = 0
    for line in test.readlines():
        if line != test_header:
            n = n + 1
            temp_line = line.replace("\n", "").split(",")
            # print(temp_line)
            # temp_record = (temp_line[0], temp_line[1], float(temp_line[2]))
            active_user = temp_line[0]
            active_item = temp_line[1]
            ## ('iCMGR_a7BuFPqZBw-IPNiA', 'crstB-H5rOfbXhV8pX0e6g', 5.0)  user, business, rate
            if active_item not in dict_bID_uIDlist:  ## not in indicates this active item is the new one
                prediction = dict_uID_ave[active_user]  ## give the ave of this user to this new item
            else:
                if active_user not in dict_uID_bIDlist:  ## not in indicates this active user the new one
                    prediction = 3.5  ## or prediction == ave of this item
                else:
                    similar_user_list = dict_bID_uIDlist[active_item]
                    prediction_number_sum = 0
                    prediction_divide_sum = 0
                    for user in similar_user_list:
                        if user != active_user:
                            corated_item = dict_uID_bIDlist[user] & dict_uID_bIDlist[active_user]
                            weight_number_sum = 0
                            weight_divide_active_sum = 0
                            weight_divide_user_sum = 0
                            if len(corated_item) == 0:
                                temp_weight = 0
                            else:
                                for each_item in corated_item:
                                    weight_number_sum = weight_number_sum + (
                                                dict_uIDandbID_rate[(active_user, each_item)] - dict_uID_ave[
                                            active_user]) * (dict_uIDandbID_rate[(user, each_item)] - dict_uID_ave[
                                        user])
                                    weight_divide_active_sum = weight_divide_active_sum + (
                                                dict_uIDandbID_rate[(active_user, each_item)] - dict_uID_ave[
                                            active_user]) ** 2
                                    weight_divide_user_sum = weight_divide_user_sum + (
                                                dict_uIDandbID_rate[(user, each_item)] - dict_uID_ave[user]) ** 2
                                # temp_weight = weight_number_sum/(pow(weight_divide_active_sum, 0.5)*pow(weight_divide_user_sum, 0.5))

                                if (pow(weight_divide_active_sum, 0.5) * pow(weight_divide_user_sum, 0.5)) != 0:
                                    temp_weight = weight_number_sum / (
                                                pow(weight_divide_active_sum, 0.5) * pow(weight_divide_user_sum, 0.5))
                                else:
                                    temp_weight = 0

                            prediction_number_sum = prediction_number_sum + temp_weight * (
                                        dict_uIDandbID_rate[(user, active_item)] - dict_uID_ave[user])
                            prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                    if prediction_divide_sum != 0:
                        prediction = dict_uID_ave[active_user] + prediction_number_sum / prediction_divide_sum
                    else:
                        prediction = dict_uID_ave[active_user]

            record = (active_user, active_item, prediction)
            fileObject.write(str(record).strip("()").replace("'", ""))
            fileObject.write("\n")

            sum_difference = sum_difference + (prediction - float(temp_line[2])) ** 2
    MSE = sum_difference / n
    RMSE = pow(MSE, 0.5)
    print(RMSE)

    fileObject.close()

    end = time.perf_counter()
    print("duration:", end - start)

if case == 3:

    start = time.perf_counter()

    train = open(train_filepath, 'r')
    train_header = train.readline()

    dict_bID_uIDlist = {}
    # 't_xXPh62lfkPc_wgF-Iasg': ['n5kvsYIIPUVrARu0pkaazQ', 'F_0Rf6KGokgemEBep0v3Tg', 'pN6pzJR6mK7549M0azoaxg']

    dict_uID_sumrating_count = {}
    #  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]

    dict_uID_ave = {}
    # 'DlXeHPut0_ZYBJXNRFjuEA': 4.238095238095238,

    dict_bID_sumrating_count = {}
    #  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]

    dict_bID_ave = {}
    # 'DlXeHPut0_ZYBJXNRFjuEA': 4.238095238095238,

    dict_uID_bIDlist = {}
    # 'k0QpEqseu2y87KIHyGVlzw': ['9tfw-OEfpF0qC2hSzRks6g', 'o8MxZKmLhdrRk8EoRX8Sog']

    dict_uIDandbID_rate = {}
    #  ('8tX-EprIfJarL2n88bnqkw', 'UvFRZR-i7p21YQVoB-8f1g'): 5.0

    for line in train.readlines():
        # print(line)
        if line != train_header:
            temp_line = line.replace("\n", "").split(",")
            # print(temp_line)

            if temp_line[1] in dict_bID_uIDlist:
                dict_bID_uIDlist[temp_line[1]].add(temp_line[0])
            else:
                dict_bID_uIDlist[temp_line[1]] = set()
                dict_bID_uIDlist[temp_line[1]].add(temp_line[0])
            # print(dict_bID_uIDlist)

            if temp_line[0] in dict_uID_bIDlist:
                dict_uID_bIDlist[temp_line[0]].add(temp_line[1])
            else:
                dict_uID_bIDlist[temp_line[0]] = set()
                dict_uID_bIDlist[temp_line[0]].add(temp_line[1])

            dict_uIDandbID_rate[(temp_line[0], temp_line[1])] = float(temp_line[2])

            if temp_line[0] in dict_uID_sumrating_count:
                dict_uID_sumrating_count[temp_line[0]][0] = dict_uID_sumrating_count[temp_line[0]][0] + float(
                    temp_line[2])
                dict_uID_sumrating_count[temp_line[0]][1] = dict_uID_sumrating_count[temp_line[0]][1] + 1
            else:
                dict_uID_sumrating_count[temp_line[0]] = [0, 0]
                dict_uID_sumrating_count[temp_line[0]][0] = dict_uID_sumrating_count[temp_line[0]][0] + float(
                    temp_line[2])
                dict_uID_sumrating_count[temp_line[0]][1] = dict_uID_sumrating_count[temp_line[0]][1] + 1

            if temp_line[1] in dict_bID_sumrating_count:
                dict_bID_sumrating_count[temp_line[1]][0] = dict_bID_sumrating_count[temp_line[1]][0] + float(
                    temp_line[2])
                dict_bID_sumrating_count[temp_line[1]][1] = dict_bID_sumrating_count[temp_line[1]][1] + 1
            else:
                dict_bID_sumrating_count[temp_line[1]] = [0, 0]
                dict_bID_sumrating_count[temp_line[1]][0] = dict_bID_sumrating_count[temp_line[1]][0] + float(
                    temp_line[2])
                dict_bID_sumrating_count[temp_line[1]][1] = dict_bID_sumrating_count[temp_line[1]][1] + 1

    for i in dict_uID_sumrating_count:
        # print(i)
        dict_uID_ave[i] = dict_uID_sumrating_count[i][0] / dict_uID_sumrating_count[i][1]

    for i in dict_bID_sumrating_count:
        # print(i)
        dict_bID_ave[i] = dict_bID_sumrating_count[i][0] / dict_bID_sumrating_count[i][1]

    fileObject = open(output_filepath, 'w')
    fileObject.write("user_id, business_id, prediction\n")

    test = open(val_filepath, 'r')
    test_header = test.readline()
    sum_difference = 0
    n = 0
    for line in test.readlines():
        if line != test_header:
            n = n + 1
            temp_line = line.replace("\n", "").split(",")
            # print(temp_line)
            # temp_record = (temp_line[0], temp_line[1], float(temp_line[2]))
            active_user = temp_line[0]
            active_item = temp_line[1]
            ## ('iCMGR_a7BuFPqZBw-IPNiA', 'crstB-H5rOfbXhV8pX0e6g', 5.0)  user, business, rate
            if active_user not in dict_uID_bIDlist:  ## not in indicates this active user is the new one
                prediction = dict_bID_ave[active_item]  ## give the ave of this user to this new item
            else:
                if active_item not in dict_bID_uIDlist:  ## not in indicates this active item the new one
                    prediction = dict_uID_ave[active_user]  ## or prediction == ave of this item
                else:
                    similar_business_list = dict_uID_bIDlist[active_user]
                    prediction_number_sum = 0
                    prediction_divide_sum = 0
                    for business in similar_business_list:
                        if business != active_item:
                            corated_user = dict_bID_uIDlist[business] & dict_bID_uIDlist[active_item]
                            weight_number_sum = 0
                            weight_divide_active_sum = 0
                            weight_divide_business_sum = 0
                            if len(corated_user) == 0:
                                temp_weight = 0
                            else:
                                for each_user in corated_user:
                                    weight_number_sum = weight_number_sum + (
                                                dict_uIDandbID_rate[(each_user, active_item)] - dict_bID_ave[
                                            active_item]) * (dict_uIDandbID_rate[(each_user, business)] - dict_bID_ave[
                                        business])
                                    weight_divide_active_sum = weight_divide_active_sum + (
                                                dict_uIDandbID_rate[(each_user, active_item)] - dict_bID_ave[
                                            active_item]) ** 2
                                    weight_divide_business_sum = weight_divide_business_sum + (
                                                dict_uIDandbID_rate[(each_user, business)] - dict_bID_ave[
                                            business]) ** 2
                                # temp_weight = weight_number_sum/(pow(weight_divide_active_sum, 0.5)*pow(weight_divide_user_sum, 0.5))

                                if (pow(weight_divide_active_sum, 0.5) * pow(weight_divide_business_sum, 0.5)) != 0:
                                    temp_weight = weight_number_sum / (
                                                pow(weight_divide_active_sum, 0.5) * pow(weight_divide_business_sum,
                                                                                         0.5))
                                else:
                                    temp_weight = 0

                            prediction_number_sum = prediction_number_sum + temp_weight * (
                                        dict_uIDandbID_rate[(active_user, business)] - dict_bID_ave[business])
                            prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                    if prediction_divide_sum != 0:
                        prediction = dict_bID_ave[active_item] + prediction_number_sum / prediction_divide_sum
                    else:
                        prediction = dict_bID_ave[active_item]

            record = (active_user, active_item, prediction)
            fileObject.write(str(record).strip("()").replace("'", ""))
            fileObject.write("\n")

            sum_difference = sum_difference + (prediction - float(temp_line[2])) ** 2
    MSE = sum_difference / n
    RMSE = pow(MSE, 0.5)
    print(RMSE)

    fileObject.close()

    end = time.perf_counter()
    print("duration:", end - start)

if case ==4:
    b = 20
    r = 2
    n = 40


    def create_signature_tuple(b_id, user_id_set):
        a_list = [271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 541, 547, 557, 563,
                  569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677,
                  929, 937, 941, 947, 953, 967]

        signature_list = []
        for i in range(1, n + 1):
            # print(i)
            # a=55+79*i
            # b=89+47*i
            # m=75+133*i
            a = a_list[i]
            b = a_list[i + 1]
            m = len(all_user_list)

            temp_index_list = []
            for user_id in user_id_set:
                temp_index = (a * dict_user[user_id] + b) % m
                temp_index_list.append(temp_index)

            signature_list.append(min(temp_index_list))
            # print()
        return tuple(signature_list)


    start = time.perf_counter()
    # n = 40
    # sys.argv[1]
    sc = SparkContext('local[*]', 'Task1')
    # file_path = train_filepath
    # file_path = sys.argv[3]
    # reviewFile_path = sys.argv[1]
    file = sc.textFile(train_filepath)
    # print(type(file))
    header = file.first()
    # file_RDD = file.filter(lambda row: row != header).map(lambda s:s.split(",")).map(lambda s: (s[1], s[0], s[2])).persist()  ##(b,u,rate)
    file_RDD = file.filter(lambda row: row != header).map(lambda s: s.split(",")).map(
        lambda s: (s[1], s[0], s[2])).persist()  ##(b,u,rating)

    # file_RDD = file_RDD.repartition(1)

    all_user_RDD = file_RDD.map(lambda s: s[1]).distinct()
    all_user_list = all_user_RDD.collect()

    business_rating_RDD = file_RDD.map(lambda s: (s[0], s[2]))

    ##build a dict (business:rating)
    dict_business_rating = {}
    for i in business_rating_RDD.collect():
        dict_business_rating[i[0]] = float(i[1])
    # print(dict_business_rating)

    ## build a dict for all user
    dict_user = {}
    for i in range(len(all_user_list)):
        temp_key = all_user_list[i]
        dict_user[temp_key] = i

    # business_user_list_RDD = file_RDD.map(lambda s: (s(0), s(1))).groupByKey().map(lambda x:(x[0], set(x[1])))   ##(b,(u1,u2))
    # print(business_user_list_RDD.take(2))
    business_user_list_RDD = file_RDD.map(lambda s: (s[0], s[1])).groupByKey().map(
        lambda x: (x[0], set(x[1])))  ##(b1,(u1,u2))

    dict_bID_uID = {}  # (b_id:set(u_id))  {'8v66rFpXa87u2jRGOn8ZfA': {'nOr3_aq60yivqgcso_iDMw'}}
    for i in business_user_list_RDD.collect():
        # print(i)
        dict_bID_uID[i[0]] = i[1]
    # print(dict_bID_uIndex)
    #
    signature_matrix = business_user_list_RDD.map(
        lambda x: (x[0], create_signature_tuple(x[0], x[1]))).persist()  # (b,(h1,h2,h3,h4))
    # print(signature_matrix.collect()[0])

    dict_eachband_bid = {}  # {(band_id,1,2):b_id}
    for i in signature_matrix.collect():
        # print(i)
        ##('gTw6PENNGl68ZPUpYWP50A', (89, 0, 69, 171, 48, 7, 8, 43, 21, 6, 17, 102, 16, 207, 186, 13, 62, 26, 5, 76, 1, 5, 12, 65, 119, 42, 61, 97, 24, 50, 2, 34, 139, 2, 30, 6, 16, 54, 18, 138))
        for j in range(0, len(i[1]), r):
            # print(j)
            band_id = j / r
            # print(band_id)
            temp_key = (band_id, i[1][j], i[1][j + 1])
            # print(temp_key)
            if temp_key in dict_eachband_bid.keys():
                dict_eachband_bid[temp_key].append(i[0])
            else:
                dict_eachband_bid[temp_key] = []
                dict_eachband_bid[temp_key].append(i[0])

    # print(len(dict_eachband_bid))

    result = set()
    result_key = set()
    candidate_pair_list = []
    dict_bID_bID_similar = {}
    single_key_set = set()
    for i in dict_eachband_bid.keys():
        if len(dict_eachband_bid[i]) >= 2:
            temp = dict_eachband_bid[i]
            pair_list = itertools.combinations(sorted(temp), 2)
            for each_pair in pair_list:
                if each_pair not in result_key:
                    # print(each_pair)
                    temp_intersection = dict_bID_uID[each_pair[0]] & dict_bID_uID[
                        each_pair[1]]  ## dict_bID_uIndex[each_pair[1]] is set
                    # print(temp_intersection)
                    # print(len(temp_intersection))
                    temp_union = dict_bID_uID[each_pair[0]] | dict_bID_uID[each_pair[1]]
                    # print(temp_union)
                    # print(len(temp_union))
                    j_similarity = len(temp_intersection) / len(temp_union)
                    # print(j_similarity)
                    # print()
                    if j_similarity > 0.5:
                        temp_result = (each_pair[0], each_pair[1], j_similarity)
                        temp_result_key = (each_pair[0], each_pair[1])
                        single_key_set.add(each_pair[0])
                        single_key_set.add(each_pair[1])
                        result.add(temp_result)
                        result_key.add(temp_result_key)
                        dict_bID_bID_similar[temp_result_key] = j_similarity

    # print(len(result))
    # print(list(result)[0])
    # print(dict_bID_bID_similar)
    # print(len(dict_bID_bID_similar))
    # print(single_key_set)

    fileObject = open(output_filepath, 'w')
    fileObject.write("user_id, business_id, prediction\n")

    test = open(val_filepath, 'r')
    test_header = test.readline()
    sum_difference = 0
    n = 0

    count_exist = 0
    count_not = 0
    for line in test.readlines():
        if line != test_header:
            n = n + 1
            temp_line = line.replace("\n", "").split(",")
            active_user = temp_line[0]
            active_business = temp_line[1]
            prediction_number_sum = 0
            prediction_divide_sum = 0
            if active_business in single_key_set:
                count_exist = count_exist + 1
                # print("y")

                for k in dict_bID_bID_similar.keys():
                    # print(k)
                    # print(k[0])
                    # print(k[1])

                    if active_business == k[0]:
                        temp_weight = dict_bID_bID_similar[k]
                        # print(temp_weight)
                        temp_rate = dict_business_rating[k[1]]
                        # print(temp_rate)
                        prediction_number_sum = prediction_number_sum + temp_weight * temp_rate
                        prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                    if active_business == k[1]:
                        temp_weight = dict_bID_bID_similar[k]
                        temp_rate = dict_business_rating[k[0]]
                        prediction_number_sum = prediction_number_sum + temp_weight * temp_rate
                        prediction_divide_sum = prediction_divide_sum + abs(temp_weight)
                if prediction_divide_sum != 0:
                    prediction = prediction_number_sum / prediction_divide_sum
                else:
                    # print("i")
                    prediction = 3.8
            else:
                count_not = count_not + 1
                # print("n")
                prediction = 3.8

            record = (active_user, active_business, prediction)
            fileObject.write(str(record).strip("()").replace("'", ""))
            fileObject.write("\n")

            sum_difference = sum_difference + (prediction - float(temp_line[2])) ** 2
    # print(count_not)
    # print(count_exist)
    MSE = sum_difference / n
    RMSE = pow(MSE, 0.5)
    print(RMSE)

    fileObject.close()

    fileObject = open("ying_cheng_explanation.txt", 'w')
    fileObject.write("Compared with the result of case3, the accuracy is reduced, but the speed is improved a lot. By using the LSH algorithm, some pairs that may be similar are found. When calculating the predicted value, the calculation amount becomes smaller and the speed becomes faster.")
    fileObject.close()

    end = time.perf_counter()
    print("duration:", end - start)





















































