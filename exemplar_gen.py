"""Purpose of this code file is to generate Exemplar sentences for 
the given source and target pairs in the dataset"""
#Importing required libraries
import pandas as pd
import numpy as np

#Importing the Tokenizer frrom TorchText
try:
    import torchtext
    from torchtext.data import get_tokenizer
    tokenizer = get_tokenizer("basic_english")
    print("Tokenizer loading success")
except:
    print("Tokenizer loading failed")



#Importing the required libraries for POS tagging
#using nltk for Pos Tagging
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')



#Read the required files
try:
    df = pd.read_csv('quora-question-pairs/train.csv')
    print("File read successful")
    print(df.head())
except:
    print("File read failed")

#Downloading StopWords
stopwords = nltk.corpus.stopwords.words('english')
print("Stopwords Downloaded")


index = 100000 #Setting how many rows of the dataset we should take
df = df[:index] 

#We handles NaN values in the dataset and remove those rows entirely
df.dropna(subset=['question1', 'question2'], inplace=True)
print('NaN value rows deleted successfully')

'''
STORING DATAX AND DATAY FOR EXEMPLAR GEN
'''
dataX = df['question1'].tolist()
dataY = df['question2'].tolist()
new_df = pd.DataFrame({'dataX': dataX, 'dataY': dataY})
new_df.head()

def dict_tokens_len(data):
    tokenized_data = []
    for sentence in data:
        tokenized_data.append(tokenizer(sentence))
    final_sentences = []
    for sentence in tokenized_data:
        final_sentences.append([sentence,len(sentence)])
    final_dict = {}
    for i in range(len(final_sentences)):
        final_dict[i] = final_sentences[i]
    return final_dict

tokenized_dataX = dict_tokens_len(dataX)
tokenized_dataY = dict_tokens_len(dataY)

#Function to get pos tags list
def get_pos_tags(tokenized_sent):
    pos_tag_list = []
    pos_tagged = nltk.pos_tag(tokenized_sent, tagset='universal')
    for i in range(len(pos_tagged)):
        pos_tag_list.append(pos_tagged[i][1])
    return pos_tag_list

pos_tagged_X = []
pos_tagged_Y = []
for i in range(len(tokenized_dataX)):
    pos_tagged_X.append(get_pos_tags(tokenized_dataX[i][0]))
    pos_tagged_Y.append(get_pos_tags(tokenized_dataY[i][0]))
    if i%10000 == 0: #Keep track of the progress
        print(i, 'sentences done')

#put pos_tagged_X and pos_tagged_Y in a new dataframe so that we can save them later
new_df['pos_tagged_X'] = pos_tagged_X
new_df['pos_tagged_Y'] = pos_tagged_Y
new_df.head()

"""Function to calculate Edit Distance"""
def edit_distance(list1, list2):
    matrix = [[0 for i in range(len(list2) + 1)] for j in range(len(list1) + 1)]
    for i in range(len(list1) + 1):
        matrix[i][0] = i
    for j in range(len(list2) + 1):
        matrix[0][j] = j
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i-1] == list2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1
    return matrix[len(list1)][len(list2)]

"""Main Function to search for Exemplar Sentences"""
def search_exemplar(tokenized_dataX, tokenized_dataY):
    c_final_set = [] #Store final generated exemplar sentences
    for i in range(len(tokenized_dataY)):
        c1_set = []
        c2_set = []
        #set it as length of pos_taggedY for i
        edit_dist_set = len(pos_tagged_Y[i])
        Y_len = 0
        for word in tokenized_dataY[i][0]:
            if word not in stopwords:
                Y_len += 1
        for j in range(len(tokenized_dataX)): #Storing sentences if they have about a similar length
            if abs(tokenized_dataY[i][1] - tokenized_dataX[j][1]) <= 2:
                c1_set.append([j, tokenized_dataX[j][0]])
        for sentence in c1_set: 
            count = 0
            for word in sentence[1]:
                if word not in stopwords and word in tokenized_dataY[i][0]:
                    count += 1
            if count <= Y_len - 2:
                c2_set.append(sentence) #c2_set is of form [[sentence number, sentence], [sentence number, sentence], ...]
        counter = 0
        short_edit_sentence = ''
        for sentence in c2_set:
            #calculate edit distance between pos tags using the Edit Distance function from above
            edit_dist = edit_distance(pos_tagged_Y[i], pos_tagged_X[sentence[0]])
            if edit_dist < edit_dist_set:
                edit_dist_set = edit_dist
                short_edit_sentence = sentence[1]
            if edit_dist == edit_dist_set:
                #random decision whether to replace or not
                if counter % 2 == 0:
                    short_edit_sentence = sentence[1]
                else:
                    short_edit_sentence = short_edit_sentence
            counter += 1
            if counter % 50000 == 0:
                print(counter, 'sentences processed')
        c_final_set.append(short_edit_sentence)
    return c_final_set

print("Generating Exemplar Sentences and storing it in a list")
c_final_set = search_exemplar(tokenized_dataX, tokenized_dataY)
#append c_final_set to new_df
new_df['c_final_set'] = c_final_set
#save new_df to a csv file for future use as the new dataset with source, target and exemplar sentences
new_df.to_csv('new_df.csv', index=False)


