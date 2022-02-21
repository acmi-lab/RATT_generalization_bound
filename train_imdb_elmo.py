import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import keras.layers as layers

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Embedding, BatchNormalization, LSTM, Dense, Concatenate
from keras.models import Model

from keras.utils import plot_model
from datasets.IMDb_elmo import * 
from keras.optimizers import adam
from itertools import tee

import re
import os
import pandas as pd
from collections import Counter
import numpy as np

import tensorflow as tf

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
  
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
          data["sentence"].append(f.read())
          data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
     
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True)
 
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
    return train_df, test_df


# Load and process the dataset files from local storage.
def download_and_load_datasets_local(force_download=False):
  
    train_df = load_dataset(os.path.join(os.path.dirname("./"), 
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname("./"), 
                                      "aclImdb", "test"))
  
    return train_df, test_df


# parameter of max word length
time_steps = 100


# building vocabulary from dataset
def build_vocabulary(sentence_list):
    unique_words = " ".join(sentence_list).strip().split()
    word_count = Counter(unique_words).most_common()
    vocabulary = {}
    for word, _ in word_count:
        vocabulary[word] = len(vocabulary)        

    return vocabulary


# Get vocabulary vectors from document list
# Vocabulary vector, Unknown word is 1 and padding is 0
# INPUT: raw sentence list
# OUTPUT: vocabulary vectors list
def get_voc_vec(document_list, vocabulary):    
    voc_ind_sentence_list = []
    for document in document_list:
        voc_idx_sentence = []
        word_list = document.split()
        
        for w in range(time_steps):
            if w < len(word_list):
                # pickup vocabulary id and convert unknown word into 1
                voc_idx_sentence.append(vocabulary.get(word_list[w], -1) + 2)
            else:
                # padding with 0
                voc_idx_sentence.append(0)
            
        voc_ind_sentence_list.append(voc_idx_sentence)
        
    return np.array(voc_ind_sentence_list)


# mini-batch generator
def train_batch_iter(data, labels, true_idx, random_idx, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("batch_size", batch_size)
    print("num_batches_per_epoch", num_batches_per_epoch)

    def data_generator():
        data_size = len(data)

        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
                shuffled_true_idx = true_idx[shuffle_indices]
                shuffled_random_idx = random_idx[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels
                shuffle_true_idx = true_idx
                shuffle_random_idx = random_idx

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                                
                X_voc = get_voc_vec(shuffled_data[start_index: end_index], vocabulary)
                                
                sentence_split_list = []
                sentence_split_length_list = []
            
                for sentence in shuffled_data[start_index: end_index]:    
                    sentence_split = sentence.split()
                    sentence_split_length = len(sentence_split)
                    sentence_split += ["NaN"] * (time_steps - sentence_split_length)
                    
                    sentence_split_list.append((" ").join(sentence_split))
                    sentence_split_length_list.append(sentence_split_length)
        
                X_elmo = np.array(sentence_split_list)

                X = [X_voc, X_elmo]
                y = shuffled_labels[start_index: end_index]
                _true_idx = shuffled_true_idx[start_index: end_index]
                _random_idx = shuffled_random_idx[start_index: end_index]
                
                yield X, y, _true_idx, _random_idx

    return num_batches_per_epoch, data_generator()


def test_batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("batch_size", batch_size)
    print("num_batches_per_epoch", num_batches_per_epoch)

    def data_generator():
        data_size = len(data)

        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                                
                X_voc = get_voc_vec(shuffled_data[start_index: end_index], vocabulary)
                                
                sentence_split_list = []
                sentence_split_length_list = []
            
                for sentence in shuffled_data[start_index: end_index]:    
                    sentence_split = sentence.split()
                    sentence_split_length = len(sentence_split)
                    sentence_split += ["NaN"] * (time_steps - sentence_split_length)
                    
                    sentence_split_list.append((" ").join(sentence_split))
                    sentence_split_length_list.append(sentence_split_length)
        
                X_elmo = np.array(sentence_split_list)

                X = [X_voc, X_elmo]
                y = shuffled_labels[start_index: end_index]
                
                yield X, y

    return num_batches_per_epoch, data_generator()

# embed elmo method
def make_elmo_embedding(x):
    embeddings = elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
    
    return embeddings


def train_acc_eval(model,train_loader, num_batches=50): 
    
    clean_correct = 0.0 
    clean_tot = 0.0 
    
    random_correct = 0.0 
    random_tot = 0.0 
    for i in range(num_batches):
        X, y, true_idx, random_idx = next(train_loader)

        preds = model.predict(X)

        output = np.squeeze(preds >0.5)

        acc = (output==y)
        clean_acc = np.multiply(true_idx, acc)
        random_acc = np.multiply(random_idx, acc)
        
        clean_correct += np.sum(clean_acc)
        clean_tot += np.sum(true_idx)
        
        random_correct += np.sum(random_acc)
        random_tot += np.sum(random_idx)
    
    return clean_correct/clean_tot, random_correct/random_tot
    
def test_acc_eval(model, valid_loader, num_batches=10): 
    
    correct = 0.0 
    tot = 0.0 
    for i in range(num_batches):
        X, y = next(valid_loader)
        preds = model.predict(X)
        
        output = np.squeeze((preds >0.5))
        
        acc = (output==y)
        correct += np.sum(acc)
        tot += len(acc)
        
    return correct/tot
        

train_df, test_df = download_and_load_datasets_local()
# print(train_df.head())

vocabulary = build_vocabulary(train_df["sentence"])


# Instantiate the elmo model
elmo_module = hub.Module("https://tfhub.dev/google/elmo/1", trainable=False)

# Initialize session
sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(1)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# elmo embedding dimension
elmo_dim = 1024

# Input Layers
word_input = Input(shape=(None, ), dtype='int32')  # (batch_size, sent_length)
elmo_input = Input(shape=(None, ), dtype=tf.string)  # (batch_size, sent_length, elmo_size)

# Hidden Layers
word_embedding = Embedding(input_dim=len(vocabulary), output_dim=128, mask_zero=True)(word_input)
elmo_embedding = layers.Lambda(make_elmo_embedding, output_shape=(None, elmo_dim))(elmo_input)
word_embedding = Concatenate()([word_embedding, elmo_embedding])
word_embedding = BatchNormalization()(word_embedding)
x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(word_embedding)


# Output Layer
predict = Dense(units=1, activation='sigmoid')(x)


model = Model(inputs=[word_input, elmo_input], outputs=predict)
# opt = keras.optimizers.Adam(learning_rate=0.01)
opt = adam(lr=0.01, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

model.summary()


# Create datasets (Only take up to time_steps words for memory)
train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text)
train_label = np.array(train_df['polarity'].tolist())

# print(train_text[0])
random_size = int(0.2*len(train_label))

idx = np.random.choice(range(len(train_label)), size =random_size,replace=False)
new_labels = np.concatenate( (np.zeros(random_size//2, dtype= np.int_), np.ones(random_size//2, dtype= np.int_) ))
np.random.shuffle(new_labels)

train_label[idx] =  new_labels
true_labels = np.ones(len(train_label))
random_labels = np.zeros(len(train_label))
random_labels[idx] = 1.0
true_labels[idx] = 0.0


test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:time_steps]) for t in test_text]
test_text = np.array(test_text)
test_label = np.array(test_df['polarity'].tolist())

# mini-batch size
batch_size = 32

train_steps, train_batches = test_batch_iter(train_text,
                                        train_label, 
                                        batch_size)


iterations = 50
check_acc = 500
copies = int(iterations//train_steps)

sampler = tee(train_batches)


test_array = []
train_array = []
bound_array = []


print(f"Steps, Train acc, Pred acc, True acc\n")

for i in range(0,iterations):
    epoch = int(i//train_steps)
    for j in range(0,check_acc):
        X, y = next(sampler[epoch])
        model.train_on_batch(X,y)
    
    
    valid_steps, valid_batches = test_batch_iter(test_text[:5000],
                                            np.array(test_df["polarity"])[:5000],
                                            500)

    train_steps_dup, train_batches_dup = train_batch_iter(train_text,
                                            train_label, 
                                            true_labels, 
                                            random_labels,
                                            500)

    test_acc = test_acc_eval(model,valid_batches)
    clean_acc, noisy_acc = train_acc_eval(model,train_batches_dup)

    pred_err = 2*(1.0-noisy_acc) + (1.0 - clean_acc)
    pred_acc = 1.0 - pred_err

    print(f"{i*check_acc}, {clean_acc*100.0}, {pred_acc*100.0}, {test_acc*100.0} \n")


        