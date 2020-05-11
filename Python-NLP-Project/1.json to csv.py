
import json
import csv
import ast

#### loading json file data into the variable data
data=[]
with open("C:\\Users\\Swathi\\Downloads\\project\\qa_Electronics.json", encoding='utf-8') as json_file:
    for i in json_file:
        c=ast.literal_eval(i)
        data.append(c)
        
 #### creating csv file for output       
data_file=open('C:\\Users\\Swathi\\Desktop\\NLP-project\\NLP\\Electronicsdata.csv', 'w')

csv_writer=csv.writer(data_file)

####writing the header to csv file
header=['questionType', 'asin', 'answerTime','unixTime', 'question', 'answerType', 'answer']
csv_writer.writerow(header)

for emp in data:
    default_list=['NaN','NaN','NaN','NaN','NaN','NaN','NaN']
    if len(emp)!=0:
        for j in emp:
            if j=='questionType':
                default_list[0]=emp[j]
            if j=='asin':
                default_list[1]=emp[j]
            if j=='answerTime':
                default_list[2]=emp[j]
            if j=='unixTime':
                default_list[3]=emp[j]
            if j=='question':
                default_list[4]=emp[j]
            if j=='answerType':
                default_list[5]=emp[j]
            if j=='answer':
                default_list[6]=emp[j]
                
        csv_writer.writerow(default_list)
data_file.close()

### Reading csv data in pandas series
import pandas as pd
dataframe=pd.read_csv('C:\\Users\\Swathi\\Desktop\\NLP-project\\NLP\\Electronicsdata.csv')
dataframe
