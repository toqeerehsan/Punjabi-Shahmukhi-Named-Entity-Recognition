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

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.compat.v1.Session(config=config))

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

################################ LOAD A PRE-TRAINED WORD EMBEDDING MODEL ###############################
print("Loading pre-trained word 2 vec embeddings...")
word_model = gensim.models.KeyedVectors.load_word2vec_format('data/punjabi_100d_14m', binary=False)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
vcb = word_model.wv.vocab
print("Shape: "+str(pretrained_weights.shape))
print("Embedding Vocab size: "+str(vocab_size))
print("Embedding size: "+str(emdedding_size))

#average_vector = np.average(pretrained_weights,axis=0, weights=None,returned=False)
#print(average_vector)
#input()
# def word2idx(word):
#     if word in word_model.wv.vocab:
#         return word_model.wv.vocab[word].index
#     else:
#         return vocab_size
# def idx2word(idx):
#     return word_model.wv.index2word[idx]

# Read train and test sentences from files
file = codecs.open("data/cleaned/Train.txt", "r", encoding="utf-8")
train_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Train.ner", "r", encoding="utf-8")
train_tags_=file.read().splitlines()

file = codecs.open("data/cleaned/Test.txt", "r", encoding="utf-8")
test_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Test.ner", "r", encoding="utf-8")
test_tags_=file.read().splitlines()

file = codecs.open("data/cleaned/Test.txt", "r", encoding="utf-8")
dev_sents_=file.read().splitlines()
file = codecs.open("data/cleaned/Test.ner", "r", encoding="utf-8")
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

# create a word to index dictionary (key is a word) and add -PAD- and -OOV- at first two indices
#word2index = {w: i + 2 for i, w in enumerate(list(words))}     # commneted
word2index = {}

# Add indices which are available in W2V and in the training set
#word2index={}
for i, w in enumerate(word_model.wv.vocab):
    word2index[w]= word_model.wv.vocab[w].index

#ADD words in the vocabulary which are in train set but not in the pretrained embeddings
#for i, w in enumerate(list(words)):
     #if w not in word2index:
         #word2index[w] = len(word2index)
         #word2index[w] = 2   # Index of unknown entry in the elmo2vec
#word2index['-PAD-'] = len(word2index)  # The special value used for padding
#word2index['-OOV-'] = len(word2index)  # The special value used for OOVs
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

#print(vcb['سیلاب-زدگان'])
#print(word2index['سیلاب-زدگان'])
#print(word2index['<UNK>'])
#input()

# length of longest sentence to add padding because model takes the sentences with same length
MAX_LENGTH = len(max(test_sentences_X, key=len))
#MAX_LENGTH = 101
print(MAX_LENGTH)  # 182
#input()
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

#Model parameters and configurations
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout
from keras.optimizers import Adam,RMSprop

# Defining model with BiLSTM
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH,)))    # input_shape=(182,)
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(Dropout(0.2))     # Testing the effect of dropout
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002), metrics=['accuracy', ignore_class_accuracy(0)])

model.summary()

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
#input()
# training of the model
print(train_sentences_X[0])
print(train_tags_y[0])

#input()
#history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=1, validation_data=(dev_sentences_X,to_categorical(dev_tags_y, len(tag2index))), shuffle = True)
history = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=64, epochs=20, validation_split=0.05, shuffle = True)
model.save("models/keras_ner_tagger_emb.model")

scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

f= open("results/output_emb.ner","w")
f_2= open("results/output_emb.conll","w")

predictions = model.predict(test_sentences_X)
results = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
print("gold first :"+test_tags_[0])
print("cand first :"+str(results[0]))

for i,tline in enumerate(test_tags_):
    tag_line = tline.strip().split(' ')

    for index in range(0,len(tag_line)):
        f.write(str(results[i][index]).replace("-PAD-","O")+" ")
        f_2.write(test_sents[i][index] +"\t"+ str(results[i][index]).replace("-PAD-", "O") + "\n")
    f.write("\n")
    f_2.write("\n")

f.close()
f_2.close()
# Run pos evaluation script
os.system("python conlltotree.py")

######################### Save History ##############################

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
