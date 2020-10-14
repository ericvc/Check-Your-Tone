import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.regularizers import L1L2
from datetime import datetime
from py.preprocessing import pre_process_sentence


# ## Read in data set used for TRAINING
# # Source: http://ai.stanford.edu/%7Eamaas/data/sentiment/
# def get_review_text(dir):
#     files = os.listdir(dir)
#     reviews = []
#     for file in files:
#         file_path = dir + file
#         with open(file_path, "rb") as f:
#             review = f.read().decode("utf-8")
#         reviews.append(review)
#     return reviews
#
#
# ## Get raw text of positive reviews
# pos_dir1 = "data\\aclImdb_v1\\aclImdb\\train\\pos\\"
# pos_dir2 = "data\\aclImdb_v1\\aclImdb\\test\\pos\\"
# positive_reviews = [*get_review_text(pos_dir1), *get_review_text(pos_dir2)]
#
# ## Get raw text of negative reviews
# neg_dir1 = "data\\aclImdb_v1\\aclImdb\\train\\neg\\"
# neg_dir2 = "data\\aclImdb_v1\\aclImdb\\test\\neg\\"
# negative_reviews = [*get_review_text(neg_dir1), *get_review_text(neg_dir2)]
#
# ## Combine all reviews into a pandas DataFrame
# all_reviews = [*positive_reviews, *negative_reviews]
# df = pd.DataFrame()
# df["sentence"] = all_reviews  # review text as column 'sentence'
# df['sentence.lower'] = [text.lower() for text in df['sentence']]  # convert to lowercase letters
# labels = np.zeros(len(all_reviews))
# labels[0:int(len(all_reviews) / 2)] = 1
# df["label"] = labels  # label value as column 'label'
#
# ## Write processed data to CSV file
# df.to_csv("data/imdb_reviews_labeled.csv")


## Shuffle rows to mix positive and negative reviews
df = pd.read_csv("data/imdb_reviews_labeled.csv").sample(frac=1, random_state=222)
sentences = df['sentence.lower'].values
y = df['label'].values


# Clean review text - takes a few moments to complete
X_processed = pre_process_sentence(sentences)


## Tokenize words - takes a few moments to complete
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_processed)
X_tokenized = tokenizer.texts_to_sequences(X_processed)


## Save fitted tokenizer to storage for use later on
filename = 'tokenizer/sentence_tokenizer_fitted.sav'
pickle.dump(tokenizer, open(filename, 'wb'))


## Dimensions for word embeddings
maxlen = 150
embedding_dim = 100


## Pad sequences (right side only) with 0s
X = pad_sequences(X_tokenized, padding='post', maxlen=maxlen)


## Get word embeddings from pre-trained model
# create function for extracting word embeddings (line-by-line) from file
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


## Use pretained word embeddings from GloVe Project
# Source: https://nlp.stanford.edu/projects/glove/
embedding_matrix = create_embedding_matrix(
    'data\\GloVe_6B\\glove.6B.300d.txt',
    tokenizer.word_index, embedding_dim)


## Check proportion of words included in pre-trained word embeddings.
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
vocab_size = len(tokenizer.word_index) + 1
print(f"{int(np.round(nonzero_elements / vocab_size, 2) * 100)}% of words are included in the pre-trained embedding.")


## Model template functions
# RNN
def create_lstm_model(learn_rate: float = 0.001, units: int = 32, n_blocks: int=5):
    # Define optimization settings
    optimizer = Adam(lr=learn_rate)

    # Initialize model
    model = Sequential()

    # Add embedding layer
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False))

    # Add LSTM blocks
    for blocks in range(n_blocks):
        label = str(blocks)
        model.add(layers.LSTM(units=units, name="lstm_"+label, return_sequences=True))
        model.add(layers.BatchNormalization(name="batch_norm_"+label))
        model.add(layers.Activation("elu", name="activation_"+label))

    model.add(layers.LSTM(units=units, name="lstm_final"))
    model.add(layers.BatchNormalization(name="batch_norm_final"))
    model.add(layers.Activation("elu", name="activation_final"))

    # Dropout layer
    model.add(layers.Dropout(0.5, name="dropout"))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid', name="output"))

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print summary
    model.summary()

    return model


# CNN
def create_conv_model(learn_rate: float = 0.001, filters: int = 32, n_blocks: int=5):
    # Define optimization settings
    optimizer = Adam(lr=learn_rate)

    # Initialize model
    model = Sequential()

    # Add embedding layer
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False))

    # Add convolution blocks
    for blocks in range(n_blocks):
        label = str(blocks)
        model.add(layers.Conv1D(filters=filters,
                                kernel_size=3,
                                padding="same",
                                #activity_regularizer=L1L2(l1=0, l2=1e-4),
                                name="conv1D_"+label))
        model.add(layers.BatchNormalization(name="batch_norm_"+label))
        model.add(layers.Activation("relu", name="activation_"+label))
        model.add(layers.MaxPool1D(pool_size=2, name="max_pool_"+label))

    # Final convolution block (w/o MaxPooling)
    model.add(layers.Conv1D(filters=filters,
                            kernel_size=3,
                            padding="same",
                            #activity_regularizer=L1L2(l1=0, l2=1e-4),
                            name="conv1D_final_block"))
    model.add(layers.BatchNormalization(name="batch_norm_final_block"))
    model.add(layers.Activation("relu", name="activation_final_block"))

    # Dropout layer
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5, name="dropout1"))
    model.add(layers.Dense(32, activation='relu', name="dense"))
    model.add(layers.Dropout(0.5, name="dropout2"))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid', name="output"))

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print summary
    model.summary()

    return model


# ## Plotting function
# plt.style.use('ggplot')
# def plot_history(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()


## Model Fitting
# CNN
cnn_model = create_conv_model(filters=16, learn_rate=1e-4, n_blocks=1)
cnn_model.fit(X, y, batch_size=8, epochs=35, validation_split=0.5, shuffle=True)
# Save as SavedModel
cnn_save_model_file = "TensorFlow Models/model_fit_conv_{:%Y%m%d_%H%M%S}".format(datetime.now())
cnn_model.save(filepath=cnn_save_model_file, overwrite=True, include_optimizer=True, save_format=None)
# Convert to TF-Lite model and save
cnn_converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
cnn_model_conv_lite = cnn_converter.convert()
open("TensorFlow Models/conv_sentiment_classifier.tflite", "wb").write(cnn_model_conv_lite)


# RNN
rnn_model = create_lstm_model(units=32, learn_rate=1e-3, n_blocks=1)  # 25,505 trainable parameters
rnn_model.fit(X, y, batch_size=16, epochs=15, validation_split=0.5, shuffle=True)
# Save as SavedModel
rnn_save_model_file = "TensorFlow Models/model_fit_rnn_{:%Y%m%d_%H%M%S}".format(datetime.now())
rnn_model.save(filepath=rnn_save_model_file, overwrite=True, include_optimizer=True, save_format=None)
# Convert to TF-Lite model and save
rnn_converter = tf.lite.TFLiteConverter.from_keras_model(rnn_model)
rnn_model_conv_lite = rnn_converter.convert()
open("TensorFlow Models/lstm_sentiment_classifier.tflite", "wb").write(rnn_model_conv_lite)


# ## EXAMPLE CODE: Load model for use on new data
#
# # load the fitted tokenizer
# loaded_tokenizer = pickle.load(open("tokenizer/sentence_tokenizer_fitted.sav", 'rb'))
# new_sentence = [
#     "i hated visiting this place more than anything in the world. it was a disaster.",
#     "this is the best movie i've ever seen so full of excitement and beautiful moments to cherish",
#     "it was ok, good, but not great. they should add more dinosaurs to make it better.",
#     "the movie was pretty good and i liked most of it, but the acting was could use some work",
#     "this is the worst product i've ever purchased. it broke within hours of use.",
#     "new research reveals the secret to being the cutest marine animal that ever existed.",
#     "global carbon emissions are down over 80 percent as climate improves for millions",
#     "congress passes legislation to protect endangered sea turtles.",
#     "reading this book was life affirming and now i have the confidence to express my best work. great job. this is the most awesome thing ever.",
#     "the lemon potatoes were disgusting and i had a bad time. overall, this place is gross. don't ever go here if you can help it."]
# new_sentence_processed = pre_process_sentence(new_sentence)
# new_sentence_tokenized = loaded_tokenizer.texts_to_sequences(new_sentence_processed)
# X_new = pad_sequences(new_sentence_tokenized, padding='post', maxlen=100)
#
# # load the fitted model
# model = tf.keras.models.load_model("TensorFlow Models/model_fit_lstm_20201008_003745")
#
# # predictions for Xnew data
# y_new = model.predict(X_new)
#
# for text, sentiment in zip(new_sentence, y_new):
#     print(f"{text}: {np.round(sentiment,3)}")
#
# # i hated visiting this place more than anything in the world. it was a disaster.: [0.003]
# # this is the best movie i've ever seen so full of excitement and beautiful moments to cherish: [0.998]
# # it was ok, good, but not great. they should add more dinosaurs to make it better.: [0.244]
# # the movie was pretty good and i liked most of it, but the acting was could use some work: [0.25]
# # this is the worst product i've ever purchased. it broke within hours of use.: [0.]
# # new research reveals the secret to being the cutest marine animal that ever existed.: [0.]
# # global carbon emissions are down over 80 percent as climate improves for millions: [0.992]
# # congress passes legislation to protect endangered sea turtles.: [1.]
# # reading this book was life affirming and now i have the confidence to express my best work. great job. this is the most awesome thing ever.: [0.985]
# # the lemon potatoes were disgusting and i had a bad time. overall, this place is gross. don't ever go here if you can help it.: [0.298]