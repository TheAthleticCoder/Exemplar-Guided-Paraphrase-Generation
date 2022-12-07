"""This file converts the ParaNMT dataset which is in a text file format
INTO a usable CSV format"""
import pandas as pd

list1=[] #Essentially for source sentences
list2=[] #Essentially for target sentences
with open('para-nmt-50m-demo/para-nmt-50m-small.txt') as f:
    for line in f:
        line=line.strip()
        line=line.split('\t')
        list1.append(line[0])
        list2.append(line[1])

df1=pd.DataFrame(list1)
df2=pd.DataFrame(list2)
#concatenate both dataframes
df=pd.concat([df1,df2],axis=1)
#rename columns
df.columns=['source','target']
#save as a csv file
df.to_csv('para-nmt-50m-demo/para-nmt-50m.csv',index=False)



