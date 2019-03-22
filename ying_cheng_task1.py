from pyspark import SparkContext
import itertools
import os
import time
import sys


b = 20
r = 2
n = 40


def create_signature_tuple (b_id, user_id_set):
    a_list=[271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,929,937,941,947,953,967]

    signature_list = []
    for i in range(1, n+1):
        # print(i)
        # a=55+79*i
        # b=89+47*i
        # m=75+133*i
        a = a_list[i]
        b = a_list[i+1]
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
# file_path = '/Users/irischeng/INF553/Assignment/hw3/yelp_train.csv'
file_path = sys.argv[1]
# file_path = sys.argv[3]
# reviewFile_path = sys.argv[1]
file = sc.textFile(file_path)
# print(type(file))
header = file.first()
file_RDD = file.filter(lambda row: row != header).map(lambda s:s.split(",")).map(lambda s: (s[1], s[0])).persist()  ##(b,u)
# file_RDD = file_RDD.repartition(1)

all_user_RDD = file_RDD.map(lambda s: s[1]).distinct()
all_user_list = all_user_RDD.collect()

## build a dict for all user
dict_user = {}
for i in range(len(all_user_list)):
    temp_key = all_user_list[i]
    dict_user[temp_key] = i

business_user_list_RDD = file_RDD.groupByKey().map(lambda x:(x[0], set(x[1])))   ##(b,(u1,u2))
# print(business_user_list_RDD.take(2))


dict_bID_uID = {}   #(b_id:set(u_id))  {'8v66rFpXa87u2jRGOn8ZfA': {'nOr3_aq60yivqgcso_iDMw'}}
for i in business_user_list_RDD.collect():
    # print(i)
    dict_bID_uID[i[0]] = i[1]
# print(dict_bID_uIndex)
#
signature_matrix = business_user_list_RDD.map(lambda x:(x[0], create_signature_tuple(x[0],x[1]))).persist()  # (b,(h1,h2,h3,h4))
# print(signature_matrix.collect()[1])

dict_eachband_bid={}   #{(band_id,1,2):b_id}
for i in signature_matrix.collect():
    # print(i)
    ##('gTw6PENNGl68ZPUpYWP50A', (89, 0, 69, 171, 48, 7, 8, 43, 21, 6, 17, 102, 16, 207, 186, 13, 62, 26, 5, 76, 1, 5, 12, 65, 119, 42, 61, 97, 24, 50, 2, 34, 139, 2, 30, 6, 16, 54, 18, 138))
    for j in range(0,len(i[1]),r):
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
# print(dict_eachband_bid[(17.0, 1503, 3463)])
# print(type())


result = set()
result_key =set()
candidate_pair_list=[]
for i in dict_eachband_bid.keys():
    if len(dict_eachband_bid[i]) >= 2:
        temp = dict_eachband_bid[i]
        pair_list = itertools.combinations(sorted(temp), 2)
        for each_pair in pair_list:
            if each_pair not in result_key:
                # print(each_pair)
                temp_intersection = dict_bID_uID[each_pair[0]] & dict_bID_uID[each_pair[1]]  ## dict_bID_uIndex[each_pair[1]] is set
                # print(temp_intersection)
                # print(len(temp_intersection))
                temp_union = dict_bID_uID[each_pair[0]] | dict_bID_uID[each_pair[1]]
                # print(temp_union)
                # print(len(temp_union))
                j_similarity = len(temp_intersection) / len(temp_union)
                # print(j_similarity)
                # print()
                if j_similarity >= 0.5:
                    temp_result = (each_pair[0], each_pair[1], j_similarity)
                    temp_result_key = (each_pair[0], each_pair[1])
                    result.add(temp_result)
                    result_key.add(temp_result_key)
print(len(result))
# print(result)


fileObject = open(sys.argv[2], 'w')
fileObject.write("business_id_1, business_id_2, similarity\n")
for i in sorted(result):
    fileObject.write(str(i).strip("()").replace("'", ""))
    fileObject.write("\n")
fileObject.close()

end = time.perf_counter()
print("duration:", end-start)

