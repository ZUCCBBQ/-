import nltk
import math
import numpy as np
import string
import xlwt
import pandas as pd
import argparse
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
from pandas.core.frame import DataFrame
from sklearn.datasets import dump_svmlight_file

# 命令行输入
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input", help="the full path of input file")
# parser.add_argument("-out", "--output", help="the full path of input file")
args = parser.parse_args()


# fcsv = open(args.output, 'a', encoding='utf-8')
def count():
    
    data = pd.read_csv(args.input)
    column = list(data['MESH_allTerms'])
    months = list(data['months'])
    
    monthword = [[]]*12
    for i in range(len(months)):
        if months[i] == 1:
            monthword[0] = monthword[0] + column[i].split(",,,")
        elif months[i] == 2:
            monthword[1] = monthword[1] + column[i].split(",,,")
        elif months[i] == 3:
            monthword[2] = monthword[2] + column[i].split(",,,")
        elif months[i] == 4:
            monthword[3] = monthword[3] + column[i].split(",,,")
        elif months[i] == 5:
            monthword[4] = monthword[4] + column[i].split(",,,")
        elif months[i] == 6:
            monthword[5] = monthword[5] + column[i].split(",,,")
        elif months[i] == 7:
            monthword[6] = monthword[6] + column[i].split(",,,")
        elif months[i] == 8:
            monthword[7] = monthword[7] + column[i].split(",,,")
        elif months[i] == 9:
            monthword[8] = monthword[8] + column[i].split(",,,")
        elif months[i] == 10:
            monthword[9] = monthword[9] + column[i].split(",,,")
        elif months[i] == 11:
            monthword[10] = monthword[10] + column[i].split(",,,")
        else:
            monthword[11] = monthword[11] + column[i].split(",,,")
    class_sort = [[]]*12
    
    for i in range(len(monthword)):
        if len(monthword[i]) == 0:
            continue
        else:
            count = Counter(monthword[i])
            count_dict = dict(count)
            class_sort[i] = sorted(count_dict.items(),
                        key=lambda item: item[1], reverse=True)
              
    w = []
    key = []
    no_mon = []
    dataframe = []
    for i in range(len(class_sort)):
        if len(class_sort[i])== 0 :
            no_mon.append(str(i+1))
        else:
            w.clear()
            key.clear()
            for j in range(len(class_sort[i])):
                w.append(class_sort[i][j][0].replace(',',' '))
                key.append(class_sort[i][j][1])
            dict1 = {"words":w,
                str(i+1):key }
            data = DataFrame(dict1)
            dataframe.append(data)
    for i in range(len(dataframe)):
        dataframe[i].set_index(['words'],inplace=True)
    result = pd.concat(dataframe,axis=1,sort=True)
    
    result=result.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12']).fillna(0.0)
    return result





if __name__ == '__main__':
    result=count()
    print(result)
    # y = result['12']  
    # dummy = pd.get_dummies(result_test.iloc[:, 1:])
    dummy_train = pd.get_dummies(result.iloc[:, [0,1,2,3,4]],columns=['1','2','3','4','5'])
    dummy_test = pd.get_dummies(result.iloc[:, [5,6,7,8,9]],columns=['6','7','8','9','10'])

    mat_train = dummy_train.as_matrix()
    mat_test = dummy_test.as_matrix()
    # mat =  dummy.as_matrix()
    dump_svmlight_file(mat_test, y, 'svm_output_test.csv') 
    dump_svmlight_file(mat_train, y, 'svm_output_train.csv') 
    print(type(result))
    result.to_csv("in_count_2017.csv")
    # print(result)    
        