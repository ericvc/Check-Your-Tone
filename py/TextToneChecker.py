import tensorflow as tf
import numpy as np
from py.preprocessing import pre_process_sentence
import pickle
from keras.preprocessing.sequence import pad_sequences


class TextToneChecker:

    def __init__(self, text):

        self.text = text
        self.tokenize_text()
        self.load_rnn()
        self.load_cnn()

    def load_tokenizer(self):

        self.tokenizer = pickle.load(open("tokenizer/sentence_tokenizer_fitted.sav", 'rb'))

    def tokenize_text(self):

        self.load_tokenizer()
        assert self.text and self.tokenizer
        # Convert text to tokenized vector
        X_processed = pre_process_sentence([self.text.lower()])
        X_tokenized = self.tokenizer.texts_to_sequences(X_processed)
        X_padded = pad_sequences(X_tokenized, padding='post', maxlen=150)
        self.text_tokenized = np.array(X_padded, dtype=np.float32)

    def load_rnn(self):

        """
        Loads RNN saved as a TensorFLow Lite model.
        """
        # Load model from file and allocate tensors
        self.rnn_model = tf.lite.Interpreter("TensorFlow Models/lstm_sentiment_classifier.tflite")
        # Allocate memory for model input tensors
        self.rnn_model.allocate_tensors()

    def load_cnn(self):

        """
        Loads RNN saved as a TensorFLow Lite model.
        """
        # Load model from file and allocate tensors
        self.cnn_model = tf.lite.Interpreter("TensorFlow Models/conv_sentiment_classifier.tflite")

    def predict(self, model_name: str):

        """
        From the transcript returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value between 0 and 1 indicating whether the input skews negative or positive.
        """
        ## Load model
        assert model_name in ["cnn", "rnn"]

        if model_name == "cnn":
            model = self.cnn_model
        elif model_name == "rnn":
            model = self.rnn_model

        ## Get model predictions
        # Allocate memory for model input tensors
        model.allocate_tensors()
        # Input vector
        input_X = self.transcript_tokenized
        # Set input and output tensors
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        # Model prediction
        model.set_tensor(input_details[0]['index'], input_X)
        model.invoke()
        pred = model.get_tensor(output_details[0]['index'])
        return pred[0][0]  # Between 0 and 1

    def predict_ensemble(self):

        self.cnn_prediction = float(self.predict(model_name="cnn"))
        self.rnn_prediction = float(self.predict(model_name="rnn"))
        return self.cnn_prediction, self.rnn_prediction
