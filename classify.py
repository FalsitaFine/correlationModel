
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
#import matplotlib.pyplot as plt

import time


PUNCTUATIONS = {".PERIOD": 1, ",COMMA": 2}
PUNCTUATIONS_RE = {0: " ", 1: ".PERIOD", 2:",COMMA"}


vocab_list = []
index_list = []

punc_index = []
word_index = []


vocab_dictionary = {}
#readvocb = open("./raw_data/vocab",'r')
readvocb = open("./web_data/vocab",'r')


#vocab_dictionary.update({"Three":3})
#print(vocab_dictionary["Three"])


index = 0
line = readvocb.readline()
while line != "":
    vocab_list.append(line.replace("\n",""))
    index_list.append(index)
    line = readvocb.readline()
    index+=1


seq_length = 1

vocab_dictionary = dict(zip(vocab_list,index_list))
#print(vocab_dictionary)
#readtrain = open("./raw_data/train.txt",'r')
readtrain = open("./web_data/fake.txt",'r')



num_punc = 0
longest_seq = 10
current_index = 0
mark_index = 0


punctuation_temp = " "
line = readtrain.readline()




## In the implement of Ottokar Tilk and Tanel Alum, the input size is based on the number of 
## words in the text, each word has a corresponding punctuation

'''
Old version, for back up

while line != "":
    line = line.replace("\n","")
    words = line.split(" ")
    for i in range(len(words)):
        #print(words[i])
        if words[i]!= "":
            if words[i] in PUNCTUATIONS:
                num_punc += 1
                mark_index = current_index
                punctuation_temp = words[i]
            else:
                punc_index.append(float(PUNCTUATIONS[punctuation_temp]))
                if words[i] in vocab_dictionary:
                    word_index.append(float(vocab_dictionary[words[i]]))
                else:
                    word_index.append(0)
                punctuation_temp = " "
        current_index += 1
'''


word_set_array = [[]]




expand_text = []
while line != "":
    line = line.replace("\n","")
    words = line.split(" ")
    for i in range(len(words)):
        #print(words[i])
        if words[i]!= "":
            expand_text.append(words[i])



    line = readtrain.readline()




for i in range(1,len(expand_text)):
        #print(words[i])
                #Find a special punctuation, remember all words before it(until last punctuation) as corresponding text.

                num_punc += 1
                #punctuation_temp = words[i]

                #if longest_seq < current_index - mark_index - 1:
                #    longest_seq = current_index - mark_index - 1
                    #mark_word = [expand_text[i-2],expand_text[i-1]]
                activate = True
                offset = 1
                words_record = 0




                if not(expand_text[i] in PUNCTUATIONS):
                    if expand_text[i-1] in PUNCTUATIONS:
                        punc_index.append(float(PUNCTUATIONS[expand_text[i-1]]))
                    else:
                        punc_index.append(0)

                    while(activate == True):
                        if(i-offset<0) or (words_record > longest_seq):
                            activate = False
                     
                        if not(expand_text[i-offset] in PUNCTUATIONS):
                            words_record += 1
                            if expand_text[i-offset] in vocab_dictionary:
                                word_set_array[current_index].append(float(vocab_dictionary[expand_text[i-offset]]))
                            else:
                                word_set_array[current_index].append(0)
                        offset = offset + 1
                    word_set_array[current_index].reverse()

                    current_index += 1
                    word_set_array.append([])



word_set_array.pop()

print(num_punc)
print(longest_seq)
print(len(punc_index))
print(len(word_set_array))
print(len(expand_text))

word_set_array_expand = []
for i in range(len(punc_index)):
    while (len(word_set_array[i]) < longest_seq):
        word_set_array[i].append(0)
    for j in range(longest_seq):
        word_set_array_expand.append(word_set_array[i][j])


'''
while(len(punc_index) % seq_length != 0):
    punc_index.append(0)
    word_index.append(0)
'''

punc_array = np.array(punc_index)
word_array = np.array(word_set_array_expand)












'''


training_file = './raw_data/train_light.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)
print("vocab_size",vocab_size)
# Parameters
learning_rate = 0.01
training_iters = 10000
display_step = 1000
n_input = 7

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases

'''




#X = np.reshape(punc_array, (int(len(punc_array)/seq_length), seq_length, 1))
X = np.reshape(word_array, (int(len(word_array)/longest_seq), longest_seq, 1))
Y = punc_array
print(X.shape,X)
print(len(Y),Y)

#for i in range(len(Y)):
#    print(Y[i])

#    print(word_set_array[i])
model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
    #keras.layers.LSTM(784, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False),
    #keras.layers.Dense(128, activation=tf.nn.relu),
    #keras.layers.Dense(10, activation=tf.nn.softmax)
    keras.layers.LSTM(250,input_shape=(longest_seq,1),return_sequences=True),
    keras.layers.LSTM(150,return_sequences=False,go_backwards = True),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)

])


#Save the trained model

saved_model = "./saved_model/lstm_model_web"
save_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=saved_model,
    save_weights_only=True)




print(X.shape)

print(Y.shape)

#print(type(X[1][0][0]))
#print(type(Y[1][0][0]))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy')



#model.build(tf.TensorShape([1,1,1]))


model.fit(X, Y, epochs=150, callbacks=[save_callback])



#Load from saved model
#tf.train.latest_checkpoint(saved_model)
#model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
#model.load_weights(tf.train.latest_checkpoint(saved_model))



test_acc = model.evaluate(X, Y)

print('Test accuracy:', test_acc)

prediction = model.predict(X)






regenerated_punc = []
current_index = 0
for i in range(len(expand_text)):
    if not (expand_text[i] in PUNCTUATIONS):
        if(current_index-1>=0) and (current_index<len(word_set_array)):
            if(expand_text[current_index-1] in PUNCTUATIONS):
                print("Detected")
                print(current_index,expand_text[current_index])
                print(X[current_index])
                #print(i+1,expand_text[i+1])
                #print(X[i+1])
            else:
                print("Ref")
                print(current_index,expand_text[current_index])
                print(X[current_index])

            print(Y[current_index])
            #Doing some bias
            prediction[current_index][1] = prediction[current_index][1]
            prediction[current_index][2] = prediction[current_index][2]




            print(prediction[current_index])
            flag = 0
            predict_punc = np.argmax(prediction[current_index])
           #predict_punc = PUNCTUATIONS_RE[prediction[current_index]]
            print(predict_punc)
            regenerated_punc.append(PUNCTUATIONS_RE[predict_punc])
        else:
            regenerated_punc.append(" ")

        current_index += 1
'''
        if (i>1 and i<len(prediction)-1):
            if (prediction[i][1] >= prediction[i-1][1]) and (prediction[i][1] >= prediction[i+1][1]):
                #this should be a .P
                predict_punc = 1
                flag = 1
                print("PPPPPPPP")
            if (prediction[i][2] >= prediction[i-1][2]) and (prediction[i][2] >= prediction[i+1][2]):
                #this should be a .P
                predict_punc = 2
                print("CCCCCCCC")
                if flag == 1:
                    if prediction[i][1] > prediction[i][2]:
                        predict_punc = 1


        if flag == 0:          
            predict_punc = 0
'''

print("End of the result")

'''
for i in range(len(punc_array)):
    predict_word = np.argmax(prediction[i, 0, 0])
    print(prediction[i, 0, 2])
    print(predict_word)
'''

# Regeneration
Regeneration_text = ''
current_index = 0
for i in range(len(expand_text)):
    if not(expand_text[i] in PUNCTUATIONS):
        #print(current_index)
        if regenerated_punc[current_index] != " ":
            Regeneration_text += (" " + regenerated_punc[current_index] + " ")
        else:
            Regeneration_text += regenerated_punc[current_index] 
        Regeneration_text += expand_text[i]
        current_index+=1


log_time = time.time()
log_name = "./test_log"+str(log_time)
log = open(log_name,'a')
log.write(Regeneration_text)