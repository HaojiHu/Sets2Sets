# Run merge_order_and_sort_by_date before this file
import numpy as np
import csv
import os
import sys

def main(argv):

    # path = '../Minnemudac/dunnhumby_50k/'
    path = argv[1]
    print('Preprocessing...')
    files = os.listdir(path)
    date_attr = 1
    mat_attr = 6
    user_attr = 11

    pid_hash = {}
    usr_oid_map = []
    file_count = 0
    usr_oid_record = {}
    for fid in range(len(files)):
        count = 0
        with open(path + files[fid], 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if count == 0:
                    count += 1
                    continue
                uid = row[user_attr]
                mid = row[mat_attr]
                str_date = row[date_attr]
                date = int(str_date)
                if mid not in pid_hash:
                    pid_hash[mid] = 1
                if uid not in usr_oid_record:
                    usr_oid_record[uid] = {}
                if date not in usr_oid_record[uid]:
                    usr_oid_record[uid][date] = []
                usr_oid_record[uid][date].append(mid)
                count += 1
        file_count += 1

    average_records = 0
    num_more_than_two_records = 0
    num_more_than_three_records = 0
    num = 0
    for uid in usr_oid_record.keys():
        num_records = len(usr_oid_record[uid].keys())

        if num_records >= 2:
            average_records += num_records
            num_more_than_two_records += 1
        if num_records >= 3:
            num_more_than_three_records += 1

    average_records = average_records / num_more_than_two_records



    print('In the ' + str() + ' month:')
    print('Total :' + str(len(usr_oid_record.keys())) + ' users')
    print('Average records: ' + str(average_records))
    print('More than one record: ' + str(num_more_than_two_records))
    print('More than two records: ' + str(num_more_than_three_records))
    usr_oid_map.append(usr_oid_record)

    num_users = 50000
    count = 0
    print('Total '+str(len(pid_hash.keys()))+' items')

    headers = ['CUSTOMER_ID','ORDER_NUMBER','MATERIAL_NUMBER']
    path = './'

    # history_file = 'Dunnhumby_history_order_original.csv'
    # history_file = 'Dunnhumby_history_order_original_10_steps.csv'
    history_file = 'Dunnhumby_history_order_original_10_steps_50kuser.csv'
    with open(path + history_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for uid in usr_oid_record.keys():
            if count > num_users:
                break
            if len(usr_oid_record[uid]) > 1:
                dates = usr_oid_record[uid].keys()
                sort_date = np.sort(list(dates))
                if len(sort_date) >= 8:
                    # for i in range(0,5):
                    for i in range(0, 5):
                        date = sort_date[i]
                        for item in usr_oid_record[uid][date]:
                            row = []
                            row.append(uid)
                            row.append(date)
                            row.append(item)
                            writer.writerow(row)
            count += 1

    count = 0
    # future_file = 'Dunnhumby_future_order_original.csv'
    # future_file = 'Dunnhumby_future_order_original_10_steps.csv'
    future_file = 'Dunnhumby_future_order_original_10_steps_50kuser.csv'
    with open(path + future_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for uid in usr_oid_record.keys():
            if count > num_users:
                break
            if len(usr_oid_record[uid]) > 1:
                dates = usr_oid_record[uid].keys()
                sort_date = np.sort(list(dates))
                if len(sort_date) >= 8:
                    for i in range(5,8):
                        date = sort_date[i]
                        for item in usr_oid_record[uid][date]:
                            row = []
                            row.append(uid)
                            row.append(date)
                            row.append(item)
                            writer.writerow(row)
            count += 1

    print('Partition the data...')
    attributes_list = ['MATERIAL_NUMBER']
    # files = ['BA_history_order_original.csv', 'BA_future_order_original.csv']
    #files = ['BA_history_order_original_100k.csv','BA_future_order_original_100k.csv']
    # files = ['BA_history_order_8kitem_200kuer.csv', 'BA_future_order_8kitem_200kuer.csv']
    # files = ['Dunnhumby_history_order_original.csv', 'Dunnhumby_future_order_original.csv']
    # files = ['Dunnhumby_history_order_original_10_steps.csv', 'Dunnhumby_future_order_original_10_steps.csv']
    files = ['Dunnhumby_history_order_original_10_steps_50kuser.csv', 'Dunnhumby_future_order_original_10_steps_50kuser.csv']

    # print('start dictionary generation...')
    # dictionary_table, num_dim, counter_table = GDF.generate_dictionary_BA(files,attributes_list)
    # print('finish dictionary generation*****')



    total_num = 0
    item_map = {}
    #data_chunk, input_size, code_freq_at_first_claim = BasketAnalysis_claim2vector.read_claim2vector_embedding_file(files)
    for file in files:
        with open(path + file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_num += 1
                if row[attributes_list[0]] not in item_map:
                    item_map[row[attributes_list[0]]] = 1
                else:
                    item_map[row[attributes_list[0]]] += 1

    import operator
    sorted_x = sorted(item_map.items(), key=operator.itemgetter(1))

    topk = 6000

    topk_num = 0
    count = 0
    topk_dictionary = {}
    for idx in range(len(sorted_x)):
        if idx >= topk:
            break
        topk_dictionary[sorted_x[-1-idx][0]] = 1
        topk_num += sorted_x[-1-idx][1]
    print('Percentage of the top '+str(topk)+' items: ' + str(topk_num/total_num))

    history = []
    future = []
    history_keys = {}
    future_keys = {}

    cus_attr = 'CUSTOMER_ID'
    with open(path + files[0], 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[attributes_list[0]] in topk_dictionary:
                instance = []
                for key in row.keys():
                    instance.append(row[key])
                history.append(instance)
                history_keys[row[cus_attr]] = 1

    with open(path + files[1], 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[attributes_list[0]] in topk_dictionary:
                instance = []
                for key in row.keys():
                    instance.append(row[key])
                future.append(instance)
                future_keys[row[cus_attr]] = 1

    # files = ['Dunnhumby_history_order.csv', 'Dunnhumby_future_order.csv']
    # files = ['Dunnhumby_history_order_10_steps.csv', 'Dunnhumby_future_order_10_steps.csv']
    # files = ['Dunnhumby_history_order_10_steps_50kuser.csv', 'Dunnhumby_future_order_10_steps_50kuser.csv']
    files = [argv[2],argv[3]]
    headers = ['CUSTOMER_ID','ORDER_NUMBER','MATERIAL_NUMBER']
    with open(path + files[0], 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in history:
            if row[0] in  history_keys and  row[0] in  future_keys:
                writer.writerow(row)


    with open(path + files[1], 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in future:
            if row[0] in  history_keys and  row[0] in  future_keys:
                writer.writerow(row)

    print('DONE!')

if __name__ == '__main__':
    main(sys.argv)