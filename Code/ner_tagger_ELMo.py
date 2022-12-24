# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import keras as krs
from keras import backend as K
import codecs
import gensim
print ("Punjabi NER tagger in Keras... :-)")

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#K.set_session(tf.Session(config=config))

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy

def append_to_list(lines):
    newlist=[]
    for i, line in enumerate(lines):
        newlist.append(lines[i].strip().split(' '))
    return newlist

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences

# Read train and test sentences from files
file = codecs.open("data/cleaned/Train.txt", "r", encoding="utf-8")
train_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Train-IO.ner", "r", encoding="utf-8")
train_tags_=file.read().splitlines()

file = codecs.open("data/cleaned/Test.txt", "r", encoding="utf-8")
test_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Test-IO.ner", "r", encoding="utf-8")
test_tags_=file.read().splitlines()

file = codecs.open("data/cleaned/Test.txt", "r", encoding="utf-8")
dev_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Test-IO.ner", "r", encoding="utf-8")
dev_tags_=file.read().splitlines()

# convert each sentence and sequence of tags to a List of words/tags
train_sents, test_sents, dev_sents, train_tags, test_tags,dev_tags \
    =append_to_list(train_sents_),append_to_list(test_sents_),append_to_list(dev_sents_), \
     append_to_list(train_tags_),append_to_list(test_tags_),append_to_list(dev_tags_)

print(train_sents[0])
print(train_tags[0])

print(test_sents[0])
print(test_tags[0])

print(dev_sents[0])
print(dev_tags[0])

#input()

print("Train sentenes: "+str(len(train_sents)))
print("Test sentenes: "+str(len(test_sents)))
print("Dev sentenes: "+str(len(dev_sents)))

print("---------------------------------------------------------------------------------------------------")
#input()

# create list if unique words and tags
words, tags = set([]), set([])
for s in train_sents:
    for w in s:
        words.add(w)

for ts in train_tags:
    for t in ts:
        tags.add(t)

############################################## LOADING Pre-training Word2vec ########################
print("Loading pre-trained word 2 vec embeddings...")
word_model = gensim.models.KeyedVectors.load_word2vec_format('data/punjabi_100d_14m', binary=False)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
vcb = sorted(word_model.wv.vocab)
print("Shape: "+str(pretrained_weights.shape))
print("Embedding Vocab size: "+str(vocab_size))
print("Embedding size: "+str(emdedding_size))

word2index= {}
for i, w in enumerate(word_model.wv.vocab):
    word2index[w]= word_model.wv.vocab[w].index

# dictionry having POS tags and -PAD-
print("Total Vocab size: "+str(len(word2index)))

tag2index = {t: i + 1 for i, t in enumerate(sorted(list(tags)))}
tag2index['-PAD-'] = 0  # The special value used to padding

# train and test sentences to have number rather than words and tags for training
train_sentences_X, test_sentences_X, dev_sentences_X, train_tags_y, test_tags_y, dev_tags_y = [], [], [], [],[],[]
for s in train_sents:
    s_int = []
    for w in s:
        if w not in word2index:
            s_int.append(word2index['-OOV-'])
        else:
            s_int.append(word2index[w])
        #try:
        #    s_int.append(word2index[w])
        #except KeyError:
        #    s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sents:
    s_int = []
    for w in s:
        if w not in word2index:
            s_int.append(word2index['-OOV-'])
        else:
            s_int.append(word2index[w])
        #try:
        #    s_int.append(word2index[w])
        #except KeyError:
        #    s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in dev_sents:
    s_int = []
    for w in s:
        if w not in word2index:
            s_int.append(word2index['-OOV-'])
        else:
            s_int.append(word2index[w])
        #try:
        #    s_int.append(word2index[w])
        #except KeyError:
        #    s_int.append(word2index['-OOV-'])

    dev_sentences_X.append(s_int)


for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

for s in dev_tags:
    dev_tags_y.append([tag2index[t] for t in s])


print(train_sentences_X[0])
print(test_sentences_X[0])
print(dev_sentences_X[0])

print(train_tags_y[0])
print(test_tags_y[0])
print(dev_tags_y[0])


# length of longest sentence to add padding because model takes the sentences with same length
MAX_LENGTH = len(max(test_sentences_X, key=len))
#MAX_LENGTH = 101
print(MAX_LENGTH)  # 182
#input()

MAX_CHAR_LENGTH = 20 # number of chars for each word
chars = set([w_i for line in train_sents for w in line for w_i in w])
n_chars = len(chars)
print("Total number of characters: ",n_chars)
char2index = {c: i + 2 for i, c in enumerate(sorted(chars))}
char2index["-OOV-"] = 1
char2index["-PAD-"] = 0
#print(len(char2index))

train_chars = []
for sentence in train_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                word_seq.append(char2index.get(sentence[i][0][j]))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    train_chars.append(np.array(sent_seq))

test_chars = []
for sentence in test_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                if sentence[i][0][j] in char2index:
                    word_seq.append(char2index.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2index.get("-OOV-"))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    test_chars.append(np.array(sent_seq))

dev_chars = []
for sentence in dev_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                if sentence[i][0][j] in char2index:
                    word_seq.append(char2index.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2index.get("-OOV-"))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    dev_chars.append(np.array(sent_seq))

# reshaping char vectors for training and testing
train_chars_X = np.array(train_chars).reshape((len(train_chars), MAX_LENGTH, MAX_CHAR_LENGTH))
test_chars_X = np.array(test_chars).reshape((len(test_chars), MAX_LENGTH, MAX_CHAR_LENGTH))
dev_chars_X = np.array(dev_chars).reshape((len(dev_chars), MAX_LENGTH, MAX_CHAR_LENGTH))


# Addign padding to train and test sentences
from keras.preprocessing.sequence import pad_sequences
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
dev_sentences_X = pad_sequences(dev_sentences_X, maxlen=MAX_LENGTH, padding='post')

train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
dev_tags_y = pad_sequences(dev_tags_y, maxlen=MAX_LENGTH, padding='post')

print(train_sentences_X[0])
print(test_sentences_X[0])
print(dev_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
print(dev_tags_y[0])

print("----------------------------------------------------------------------------------------------------")
##############################################################################
############################# ELMO settings ####################################
##############################################################################
sent_length = MAX_LENGTH
elmo_dim = 128
elmo_layer = 2

new_train_sents=[]
for i,j in enumerate(train_sents_):
    new_train_sents.append(j.strip().split(' ')[:sent_length])

new_test_sents=[]
for i,j in enumerate(test_sents_):
    new_test_sents.append(j.strip().split(' ')[:sent_length])

new_dev_sents=[]
for i,j in enumerate(dev_sents_):
    new_dev_sents.append(j.strip().split(' ')[:sent_length])

print("Loading pre-trained ELMO embeddings...")
from allennlp.commands.elmo import ElmoEmbedder
#elmo = ElmoEmbedder(options_file='/home/toqeer/PycharmProjects/train_elmo/bilm-tf-master/punjabi_data/trained_folder/options_final.json',weight_file='/home/toqeer/PycharmProjects/train_elmo/bilm-tf-master/punjabi_data/trained_folder/weights_final.hdf5')
elmo = ElmoEmbedder(options_file='/home/linadmin/PycharmProjects/train_elmo/bilm-tf-master/punjabi_data/trained_folder/options_final.json',weight_file='/home/linadmin/PycharmProjects/train_elmo/bilm-tf-master/punjabi_data/trained_folder/weights_final.hdf5')
#vectors = elmo.embed_sentence(new_train_sents[0])
#elmo_train_x = np.zeros( (sent_length, elmo_dim) )
#elmo_train_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
#elmo_train_x = np.expand_dims(elmo_train_x, axis=0)

#print(elmo_train_x.shape)

#print("done")
#input()
import sys
#### ELMO vectors for training sentances
vectors = elmo.embed_sentence(new_train_sents[0])
elmo_train_x = np.zeros( (sent_length, elmo_dim) )
elmo_train_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
elmo_train_x = np.expand_dims(elmo_train_x, axis=0)
for i in range(1,len(new_train_sents)):
    vectors = elmo.embed_sentence(new_train_sents[i])
    _x = np.zeros((sent_length, elmo_dim))
    _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
    _x = np.expand_dims(_x, axis=0)
    elmo_train_x = np.append(elmo_train_x,_x,axis=0)
    if (i % 10) == 0:
        sys.stdout.write('Train: [%d%%]\r' % int(round((i / len(new_train_sents)) * 100)))
        sys.stdout.flush()
print('Train sentences '+str(elmo_train_x.shape))

#### ELMO vectors for test sentances
vectors = elmo.embed_sentence(new_test_sents[0])
elmo_test_x = np.zeros( (sent_length, elmo_dim) )
elmo_test_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
elmo_test_x = np.expand_dims(elmo_test_x, axis=0)
for i in range(1,len(new_test_sents)):
    vectors = elmo.embed_sentence(new_test_sents[i])
    _x = np.zeros((sent_length, elmo_dim))
    _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
    _x = np.expand_dims(_x, axis=0)
    elmo_test_x = np.append(elmo_test_x,_x,axis=0)
    if (i % 10) == 0:
        sys.stdout.write('Test: [%d%%]\r' % int(round((i / len(new_test_sents)) * 100)))
        sys.stdout.flush()
print('Test sentences '+str(elmo_test_x.shape))

#### ELMO vectors for dev sentances
"""
vectors = elmo.embed_sentence(new_dev_sents[0])
elmo_dev_x = np.zeros( (sent_length, elmo_dim) )
elmo_dev_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
elmo_dev_x = np.expand_dims(elmo_dev_x, axis=0)
for i in range(1,len(new_dev_sents)):
    vectors = elmo.embed_sentence(new_dev_sents[i])
    _x = np.zeros((sent_length, elmo_dim))
    _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
    _x = np.expand_dims(_x, axis=0)
    elmo_dev_x = np.append(elmo_dev_x,_x,axis=0)
    if (i % 10) == 0:
        sys.stdout.write('Dev: [%d%%]\r' % int(round((i / len(new_dev_sents)) * 100)))
        sys.stdout.flush()
print('Dev sentences '+str(elmo_dev_x.shape))
"""
##########################################################################################################
#########################################################################################################

print("----------------------------------------------------------------------------------------------------")

#Model parameters and configurations
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout
from keras.optimizers import Adam,RMSprop


input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=50, input_length=MAX_LENGTH)(input_0)

input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')
merged_1 = concatenate([emb_0, input_1])

LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.1)(LSTM1)
dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out1)
model = Sequential()
model = Model(inputs=[input_0, input_1], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy', ignore_class_accuracy(0)])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("1BiLSTM + 50 emb + ELMo...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights])(input_0)
input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')
merged_1 = concatenate([emb_0, input_1])
LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.2)(LSTM1)
dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out1)
model = Sequential()
model = Model(inputs=[input_0, input_1], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy', ignore_class_accuracy(0)])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("1BiLSTM + 100d w2v + ELMo...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights])(input_0)
input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')

merged_1 = concatenate([emb_0, input_1])
LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.2)(LSTM1)
LSTM2= Bidirectional(LSTM(300, return_sequences=True))(drop_out1)
drop_out2 = Dropout(0.2)(LSTM2)
dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out2)
model = Sequential()
model = Model(inputs=[input_0, input_1], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy', ignore_class_accuracy(0)])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("2BiLSTM + 100d w2v + ELMo ...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights])(input_0)
input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')

merged_1 = concatenate([emb_0, input_1])
LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.2)(LSTM1)
LSTM2= Bidirectional(LSTM(300, return_sequences=True))(drop_out1)
drop_out2 = Dropout(0.2)(LSTM2)
LSTM3= Bidirectional(LSTM(300, return_sequences=True))(drop_out2)
drop_out3 = Dropout(0.2)(LSTM3)

dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out3)
model = Sequential()
model = Model(inputs=[input_0, input_1], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy', ignore_class_accuracy(0)])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("3BiLSTM + 100d w2v + ELMo ...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights])(input_0)
input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')

char_in = Input(shape=(MAX_LENGTH, MAX_CHAR_LENGTH,))
emb_char = TimeDistributed(Embedding(input_dim=len(char2index), output_dim=30, input_length=MAX_CHAR_LENGTH, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=30, return_sequences=False,recurrent_dropout=0.5))(emb_char)


merged_1 = concatenate([emb_0, input_1,char_enc])
LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.2)(LSTM1)
#LSTM2= Bidirectional(LSTM(300, return_sequences=True))(drop_out1)
#drop_out2 = Dropout(0.2)(LSTM2)
dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out1)
model = Sequential()
model = Model(inputs=[input_0, input_1,char_in], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy'])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x,train_chars_X], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x,test_chars_X],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x,test_chars_X], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x,test_chars_X])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("1BiLSTM + 100d w2v + ELMo + char emb ...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
input_0 = Input(shape=(MAX_LENGTH,))    # input_shape=(2??,)
emb_0= Embedding(input_dim=len(word2index), output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights])(input_0)
input_1 = Input(shape=(MAX_LENGTH, elmo_dim),dtype='float32')

char_in = Input(shape=(MAX_LENGTH, MAX_CHAR_LENGTH,))
emb_char = TimeDistributed(Embedding(input_dim=len(char2index), output_dim=30, input_length=MAX_CHAR_LENGTH, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=30, return_sequences=False,recurrent_dropout=0.5))(emb_char)


merged_1 = concatenate([emb_0, input_1,char_enc])
LSTM1= Bidirectional(LSTM(300, return_sequences=True))(merged_1)
drop_out1 = Dropout(0.2)(LSTM1)
LSTM2= Bidirectional(LSTM(300, return_sequences=True))(drop_out1)
drop_out2 = Dropout(0.2)(LSTM2)
dense_out = TimeDistributed(Dense(len(tag2index),activation='softmax'))(drop_out2)
model = Sequential()
model = Model(inputs=[input_0, input_1,char_in], outputs=[dense_out])
#model = Model(inputs=[input_1,char_in], outputs=[dense_out])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy'])

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
history = model.fit([train_sentences_X, elmo_train_x,train_chars_X], to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=12, validation_data=([test_sentences_X, elmo_test_x,test_chars_X],to_categorical(test_tags_y, len(tag2index))), shuffle = True)
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_ELMo.model")

scores = model.evaluate([test_sentences_X, elmo_test_x,test_chars_X], to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/ner_ELMo_output.conll","w")
predictions = model.predict([test_sentences_X, elmo_test_x,test_chars_X])
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        #f.write(str(results[i][index]).replace("-PAD-","NN")+" ")
        f.write(test_sents[i][index] + " POS "+test_tags[i][index]+" "+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
f.close()
print("2BiLSTM + 100d w2v + ELMo + char emb ...")
# Run pos evaluation script
os.system("perl results/ner/bin/conlleval -r < results/ner_ELMo_output.conll")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
######################### Save History ##############################
"""
print(history.history.keys())
out_file = open("models/ner_tagger_emb.history",'w')
out_file.write("Train Accuracy:\n")
out_file.write(str(history.history['acc']))
out_file.write("\nVal Accuracy:\n")
out_file.write(str(history.history['val_acc']))

out_file.write("\nTrain Loss:\n")
out_file.write(str(history.history['loss']))
out_file.write("\nVal Loss:\n")
out_file.write(str(history.history['val_loss']))
out_file.close()
"""
######################### DRAW GRAPHS################################
#history_dict = history.history
#print(history_dict.keys())
#print("Val LOSS")
#print(history_dict['val_loss'])
#print("Val ACCURACY")
#print(history_dict['val_ignore_accuracy'])


# import matplotlib.pyplot as plt
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
# acc_values = history_dict['ignore_accuracy']
# val_acc_values = history_dict['val_ignore_accuracy']
# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
##################### END graphs ###########################
