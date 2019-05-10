from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
import json
import sys
import time

start = time.perf_counter()

input_path = sys.argv[1]
# output_filepath = 'output.csv'
output_filepath = sys.argv[3]
# business_filepath = '/Users/irischeng/INF553/Assignment/Competition/business.json'
business_filepath = input_path + "business.json"

# train_filepath = '/Users/irischeng/INF553/Assignment/Competition/yelp_train.csv'
train_filepath = input_path + 'yelp_train.csv'

reader = Reader(rating_scale=(1, 5), line_format='user item rating', sep=',', skip_lines=1)

data = Dataset.load_from_file(train_filepath, reader=reader)

# trainset, valset = train_test_split(data, test_size=.25)
trainset = data.build_full_trainset()


# val_filepath = '/Users/irischeng/INF553/Assignment/Competition/yelp_val.csv'
val_filepath = input_path + 'yelp_val.csv'

# test_file_name = sys.argv[2]
# test_filepath = input_path + test_file_name
test_filepath = sys.argv[2]


dict_bID_uIDlist = {}

# dict_bID_sumrating_count = {}
#  'ahXJ4DktihtIc7emzuyK7g': [76.0, 20]

# dict_bID_ave = {}

train = open(train_filepath, 'r')
train_header = train.readline()

for line in train.readlines():
    # print(line)
    if line != train_header:
        temp_line = line.replace("\n", "").split(",")
        # print(temp_line)

        if temp_line[1] in dict_bID_uIDlist:
            dict_bID_uIDlist[temp_line[1]].append(temp_line[0])
        else:
            dict_bID_uIDlist[temp_line[1]] = []
            dict_bID_uIDlist[temp_line[1]].append(temp_line[0])

        # if temp_line[1] in dict_bID_sumrating_count:
        #     dict_bID_sumrating_count[temp_line[1]][0] = dict_bID_sumrating_count[temp_line[1]][0] + float(
        #         temp_line[2])
        #     dict_bID_sumrating_count[temp_line[1]][1] = dict_bID_sumrating_count[temp_line[1]][1] + 1
        # else:
        #     dict_bID_sumrating_count[temp_line[1]] = [0, 0]
        #     dict_bID_sumrating_count[temp_line[1]][0] = dict_bID_sumrating_count[temp_line[1]][0] + float(
        #         temp_line[2])
        #     dict_bID_sumrating_count[temp_line[1]][1] = dict_bID_sumrating_count[temp_line[1]][1] + 1

# for i in dict_bID_sumrating_count:
#     # print(i)
#     dict_bID_ave[i] = dict_bID_sumrating_count[i][0] / dict_bID_sumrating_count[i][1]


# print(dict_bID_uIDlist)


dict_bid_Stars = {}
businessFile = open(business_filepath,'r')

for line in businessFile:
    temp = json.loads(line)
    temp_key = temp["business_id"]
    # print(temp_key)

    temp_value = temp['stars']
    dict_bid_Stars[temp_key] = temp_value
# print(dict_bid_reviewAndStars)


svd = SVD(n_factors=20, lr_all=0.009, reg_all=0.17, n_epochs=25, random_state=1).fit(trainset)

svd.fit(trainset)


fileObject = open(output_filepath, 'w')
test = open(test_filepath, 'r')
test_header = test.readline()
sum_difference = 0
n = 0
diff_count =0
fileObject.write('user_id, business_id, prediction\n')
for line in test.readlines():
    if line != test_header:
        n = n + 1
        temp_line = line.replace("\n", "").split(",")
        # print(temp_line)
        active_user = temp_line[0]
        active_item = temp_line[1]
        true_rating = (temp_line[2].strip("'"))
        # print(true_rating)
        # print(type(true_rating))
        temp_pred = svd.predict(active_user, active_item, verbose=False).est

        if (active_item not in dict_bID_uIDlist.keys()):
            temp_pred = dict_bid_Stars[active_item]
        #     # temp_pred = dict_bID_ave[active_item]
        elif (len(dict_bID_uIDlist[active_item])<=3):
            temp_pred = dict_bid_Stars[active_item]
        # else:
        # print(temp_pred.est)
        # difference = (float(temp_pred.est)- float(temp_line[2]))**2
        # if difference >=4:
        #     diff_count = diff_count+1
        result = (active_user, active_item, temp_pred)
        fileObject.write(str(result).strip("()").replace("'", ""))
        fileObject.write("\n")
        sum_difference = sum_difference + (float(temp_pred) - float(temp_line[2])) ** 2

MSE = sum_difference / n
fileObject.close()

RMSE = pow(MSE, 0.5)
print(RMSE)
# print(n)

end = time.perf_counter()
print("duration:", end - start)
