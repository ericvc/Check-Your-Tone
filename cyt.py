import tensorflow as tf
import numpy as np
from py.preprocessing import pre_process_sentence
import pickle
from keras.preprocessing.sequence import pad_sequences
import sys
import os

class ToneChecker:

    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        self.tokenize_text()
        self.load_rnn()
        self.load_cnn()

    def tokenize_text(self):
        assert self.text and self.tokenizer
        # Convert text to tokenized vector
        X_processed = pre_process_sentence([self.text.lower()])
        X_tokenized = self.tokenizer.texts_to_sequences(X_processed)
        X_padded = pad_sequences(X_tokenized, padding='post', maxlen=100)
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

    def predict_rnn(self):
        """
        From the text returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
        """
        input_X = self.text_tokenized

        ## RNN predictions
        # Allocate memory for model input tensors
        self.rnn_model.allocate_tensors()
        # Get input and output tensor information
        input_details = self.rnn_model.get_input_details()
        output_details = self.rnn_model.get_output_details()
        # Model prediction
        tc.rnn_model.set_tensor(input_details[0]['index'], input_X)
        tc.rnn_model.invoke()
        tc.rnn_prediction = tc.rnn_model.get_tensor(output_details[0]['index'])
        return self.rnn_prediction[0][0]  # Between 0 and 1

    def predict_cnn(self):
        """
        From the text returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
        """
        input_X = self.text_tokenized

        ## CNN predictions
        # Allocate memory for model input tensors
        self.cnn_model.allocate_tensors()
        # Get input and output tensor information
        input_details = self.cnn_model.get_input_details()
        output_details = self.cnn_model.get_output_details()
        # Model prediction
        self.cnn_model.set_tensor(input_details[0]['index'], input_X)
        self.cnn_model.invoke()
        self.cnn_prediction = self.cnn_model.get_tensor(output_details[0]['index'])
        return self.cnn_prediction[0][0]  # Between 0 and 1

    def predict_ensemble(self):
        p_rnn = self.predict_rnn()
        p_cnn = self.predict_cnn()
        self.ensemble_prediction = np.mean([p_rnn, p_cnn])
        return self.ensemble_prediction


# Load word tokenizer
tokenizer = pickle.load(open("tokenizer/sentence_tokenizer_fitted.sav", 'rb'))

# Console window width
width = os.get_terminal_size().columns

print("CHECK YOUR TONE! - Text Sentiment Analysis\nReturns score between 0 (negative) and 1 (positive)\n".center(width))

try:

    # Main program loop
    while True:

        text = input("Enter some text: ")
        tc = ToneChecker(text, tokenizer=tokenizer)
        predicted_sentiment = np.round(tc.predict_ensemble(),2)

        if predicted_sentiment <= 0.33:
            print(f"\nPredicted sentiment score is {predicted_sentiment}: NEGATIVE\n".center(width))
        elif 0.33 < predicted_sentiment <= 0.67:
            print(f"\nPredicted sentiment score is {predicted_sentiment}: NEUTRAL\n".center(width))
        else:
            print(f"\nPredicted sentiment score is {predicted_sentiment}: POSITIVE\n".center(width))


except KeyboardInterrupt:
    print("Check Your Tone! closed using keyboard exit command.")

except:
    print("An error has occurred.")

finally:
    sys.exit(0)