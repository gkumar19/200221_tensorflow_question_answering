# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:03:19 2020

@author: KGU2BAN
"""
import tensorflow.keras.layers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re

from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dot, Add, Dense, Activation, LSTM, Lambda, Multiply, Flatten, GRU, Concatenate, dot, Permute, concatenate, add
from tensorflow.keras.layers import TimeDistributed, SimpleRNN, RepeatVector, Reshape, multiply, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Recall, Precision, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives 
from tensorflow.keras.models import Sequential
#define metrics
metrics = [SparseCategoricalAccuracy(name='acc')]
#%%parse data

def parse_babi2(lines, test_or_train='test', tokeniser=None):
    '''
    dataset to be structered as (story , question, answer)
    '''
    
    data = {'story':[], 'question':[], 'answer':[]}
    all_sentences = []
    for line in lines:
        line = line.strip().lower() #remove white spaces and lower case
        idx, line = line.split(' ', 1) #split into lists with max split to 1 
        idx = int(idx)
        if idx == 1: #first line of a story
            story = [] #one story which will have multiple lines
            word_list = ' '.join(re.findall(r'\w+', line)) #remove white spaces
            story.append(word_list)
            all_sentences.append(word_list)
        else:
            if '\t' not in line:
                word_list = ' '.join(re.findall(r'\w+', line))
                all_sentences.append(word_list)
                story.append(word_list)
            if '\t' in line:
                question, answer, _ = line.split('\t') #not using the supervised numbering
                question = question.replace('?', '').strip()
                data['story'].append(story.copy())
                data['question'].append(question)
                data['answer'].append(answer)
                all_sentences.append(question)
                all_sentences.append(answer)
    
    if test_or_train == 'train':
        tokeniser = Tokenizer(oov_token='oov')
        tokeniser.fit_on_texts(all_sentences)
    
    data_seq = {'story':[], 'question':[], 'answer':[]} #converted to sequence
    data_seq['question'] = pad_sequences(tokeniser.texts_to_sequences(data['question']), maxlen=3)
    data_seq['answer'] = pad_sequences(tokeniser.texts_to_sequences(data['answer']), maxlen=1)
    for i in range(len(data['story'])):
        x = pad_sequences(tokeniser.texts_to_sequences(data['story'][i]), maxlen=6)
        x = x.T
        x = pad_sequences(x, maxlen=12)
        x = x.T
        data_seq['story'].append(x)
    data_seq['story'] = np.stack(data_seq['story'], axis=0)
    x_train_test = [data_seq['story'], data_seq['question']]
    y_train_test = data_seq['answer']
    
    return tokeniser, data, x_train_test, y_train_test

with open('qa1_single-supporting-fact_train.txt') as f:
    train_lines = f.readlines() #list of lines
tokeniser, train_text, x_train, y_train = parse_babi2(train_lines, 'train')

with open('qa1_single-supporting-fact_test.txt') as f:
    test_lines = f.readlines() #list of lines
_, test_text, x_test, y_test = parse_babi2(test_lines, tokeniser=tokeniser)

#%% create model accuracy acheived 100%

tf.keras.backend.clear_session()

encode_size = 10
input_sentence_length = 6
input_question_length = 3
input_story_length = 12
num_words = len(tokeniser.word_index)

def encode_input_story(compression_type='bow', encode_size=10, input_sentence_length=6, input_story_length=12):
    '''
    model input : (batch, story_length, sentence_length)
    model output : (batch, story_length, encoding_length)
    '''
    input_1 = Input((input_story_length,input_sentence_length))
    x = Embedding(num_words+1, encode_size, input_length=input_sentence_length)(input_1)
    
    if compression_type == 'bow':
        x = Lambda(lambda x: tf.math.reduce_mean(x,axis=-2))(x)
    else:
        x = TimeDistributed(SimpleRNN(encode_size))(x)
    
    x = GRU(encode_size, return_sequences=True)(x)
    model = Model(inputs=input_1, outputs=x)
    return model

def encode_input_question(compression_type='bow', encode_size=10, input_sentence_length=3):
    '''
    model input : (batch, sentence_length)
    model output : (batch, encoding_length)
    '''
    if compression_type == 'bow':
        model = Sequential([Input((input_sentence_length)),
                            Embedding(num_words+1, encode_size, input_length=input_sentence_length),
                            Lambda(lambda x: tf.math.reduce_mean(x,axis=-2))])
        return model
    if compression_type == 'rnn':
        model = Sequential([Input((input_sentence_length)),
                            Embedding(num_words+1, encode_size, input_length=input_sentence_length),
                            SimpleRNN(encode_size)])
        return model

def attention_dot(encoded_story, encoded_question):
    '''
    encoded story: (batch_size, stories, features)
    encoded_question: (batch_size, features)
    attention: (batch_size, stories)
    '''
    encoded_question_upscaled = RepeatVector(1)(encoded_question) #(batch_size, 1, features)
    encoded_question_upscaled = Dense(10)(encoded_question_upscaled) #(batch_size, 1, 10)
    encoded_story = Dense(10)(encoded_story) #(batch_size, stories, 10)
    attention = dot([encoded_story, encoded_question_upscaled], axes=[-1,-1]) #(batch_size, stories, 1)
    attention = Flatten()(attention) #(batch_size, stories)
    attention = Activation('softmax')(attention) #(batch_size, stories)
    return attention


def apply_attention(encoded_story, attention):
    '''
    encoded story: (batch_size, stories, features)
    attention: (batch_size, stories)
    attended: (batch_size, features)
    '''
    attention = RepeatVector(encode_size)(attention) #(batch_size,encode_size, stories)
    attention = Permute([2,1])(attention) #(batch_size,stories, encode_size)
    def lambda_function(x):
        attention_ = x[0]*x[1] #(batch_size,stories, encode_size)
        attention_reduce = tf.reduce_sum(attention_, axis=1)
        #attention_reduce = concatenate([attention, attention_], axis=-1)
        #attention_reduce = GRU(10)(attention_)
        return attention_reduce
    attended = Lambda(lambda_function, name='d')([encoded_story, attention])
    return attended


def time_encoding(input_tensor):
    constant_multiplier = tf.linspace(0.0, input_story_length-1, input_story_length)
    constant_multiplier =  tf.cast(constant_multiplier, 'int32')[:,tf.newaxis]
    input_time = tf.ones_like(input_tensor , dtype='int32') * constant_multiplier
    input_time = input_time[:,:,0]
    time_embedding = Embedding(input_story_length+1, input_tensor.shape[-1], input_length=input_story_length)(input_time)
    time_embedding = time_embedding + input_tensor
    return time_embedding


def first_memory_graph(input_story, encoded_question):
    model_story = encode_input_story('bow', encode_size = encode_size)
    encoded_story = model_story(input_story)
    attention = attention_dot(encoded_story, encoded_question)
    
    model_story2 = encode_input_story('rnn', encode_size = encode_size)
    encoded_story2 = model_story2(input_story)
    
    #encoded_story = time_encoding(encoded_story)
    #encoded_story2 = time_encoding(encoded_story2)
    
    attended = apply_attention(encoded_story2, attention)
    #concat = concatenate([attended, encoded_question])
    concat = add([attended, encoded_question]) #This also works
    dense = Dense(num_words+1, activation='softmax')(concat)
    return dense

input_story = Input((input_story_length,input_sentence_length))
input_question = Input((input_question_length))
model_question = encode_input_question('rnn', encode_size = encode_size)
encoded_question = model_question(input_question)
dense_first = first_memory_graph(input_story, encoded_question)
model = Model(inputs=[input_story, input_question], outputs=dense_first)

model.summary()
#%% record changing lr after each epoch for tensorboard
logdir = 'log'
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, embeddings_freq=1)

def scheduler(epoch):
    if epoch < 30:
        lr = 0.01
    if 30 < epoch < 100:
        lr = 0.005
    else:
        lr = 0.001
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile('adam', 'sparse_categorical_crossentropy', metrics=metrics)
model.fit(x_train, y_train, epochs=100, batch_size=80, validation_data=(x_test, y_test), callbacks=[lrs, tensorboard])


#%% confusion matrix
def plot_confusion(x_train, y_train):
    y_true = tokeniser.sequences_to_texts(y_train)
    y_pred = np.argmax(model.predict(x_train), axis=-1)[:, None]
    y_pred = tokeniser.sequences_to_texts(y_pred)
    plt.figure()
    labels = list(set(y_true))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels),
                annot=True , fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')

plot_confusion(x_train, y_train)
plot_confusion(x_test, y_test)

#%% visualise attention
from tensorflow.keras.backend import function
f = function(model.inputs, model.get_layer('activation').output)
attention_weights = f(x_train)
sns.heatmap(attention_weights[:20], cmap='Blues')

#%% trying our own examples
story = ['mary went to kitchen', 'john is in kitchen', 'john journeyed to the kitchen', 'daniel went to hallway', 'mary travelled to the bathroom']
question = ['where is john']

x1 = tokeniser.texts_to_sequences(story)
x1 = pad_sequences(x1, 6)
x1 = pad_sequences(x1.T, 12).T[None,:]
x2 = pad_sequences(tokeniser.texts_to_sequences(question), 3)
predict = model.predict([x1, x2])
argsort = np.flip(np.argsort(predict)).flatten()[:3]
print(f'Guess: {tokeniser.index_word[argsort[0]]}')


#%% one more implementation with 97% accuracy, inspired from keras example on keras website
num_embedding = 10
num_question_length = 6

#defining the layers
embedding_question0 = Embedding(21, num_embedding, input_length=(num_question_length,), name='embedding_question0')
embedding_story0 = Embedding(21, num_embedding, input_length=(6,), name='embedding_story0')
embedding_story1 = Embedding(21, 6, input_length=(6,), name='embedding_story1')
#dot_attention = Dot(name='attention', axes=-1)
#gru_story0 = GRU(num_embedding, name='gru_story0', return_sequences=True)
#gru_output = GRU(num_embedding, name='gru_output', return_sequences=False)

#defining the flow
input_story = Input(shape=(12, 6), name='input_story') #none,12,6 --> 12 stories
story0 = embedding_story0(input_story) #none,12,6,5 -->5 embedding features
story0 = Lambda(lambda x: tf.math.reduce_mean(x,axis=-2) ,name='glb_avg_story0')(story0) #none,12,5
#story0 = gru_story0(story0) #none, 12, 5

story1 = embedding_story1(input_story) #none,12,6,5 -->5 embedding features
story1 = Lambda(lambda x: tf.math.reduce_mean(x, axis=-2) ,name='glb_avg_story1')(story1) #none,12,5

input_question = Input(shape=(num_question_length,), name='input_question') #None,6
question0 = embedding_question0(input_question) #none,6,5
#question0 = Lambda(lambda x: tf.math.reduce_mean(x, axis=-2, keepdims=True), name='lambda_question')(question0) #none,1,5
#flattened_question = Flatten(name='flatten_question')(question0)

#attention = dot_attention([story0, question0]) #none, 12, 1
attention = dot([story0, question0], axes=(2, 2)) # none, 12, 6
attention = Activation('softmax', name='attention')(attention) # 

#attention = Lambda(lambda x: tf.keras.activations.softmax(x, axis=-2), name='softmax')(attention) #none,12,1
#attention_applied = Multiply(name='attenstion_applied')([attention, story1]) #none,12,5
#attention_applied = GlobalAveragePooling1D(name='glb_avg_story_attenstion')(attention_applied) #none, 5

#output = Add(name='addition')([attention_applied, flattened_question]) #none,5
output = add([attention, story1]) # (samples, story_maxlen, query_maxlen)
output = Permute((2, 1))(output)
output = concatenate([output, question0])
output = LSTM(32)(output)
output = Dense(21,activation='softmax', name='dense_output')(output) #none,1


#learning rate finder
def scheduler(epoch):
    if epoch < 50:
        lr = 0.0015*np.e*(epoch*0.08) #for scheduling
    else:
        lr = 0.0008*np.e*(epoch*0.077)
    return lr

lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = Model(inputs=[input_story, input_question], outputs=output)
model.compile('adam', 'sparse_categorical_crossentropy', metrics=metrics)
model.summary()

x_train_double = x_train.copy()
x_train_double[1] = np.tile(x_train_double[1], [1,2])
logdir = 'log'
tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, embeddings_freq=1)
model.fit(x_train_double, y_train, epochs=300, callbacks=[tensorboard, lrs], batch_size=200)
