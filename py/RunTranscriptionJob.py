import boto3
import time
import urllib
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
from py.preprocessing import pre_process_sentence
import pickle
from keras.preprocessing.sequence import pad_sequences


## AWS Authentication Settings
with open("/home/pi/Projects/Check-Your-Tone/amazon_tokens.json") as f:
    keys = json.load(f)

AWS_ACCESS_KEY_ID = keys["ACCESS"]
AWS_SECRET_ACCESS_KEY = keys["ACCESS_SECRET"]


class RunTranscriptionJob:
    def __init__(self, bucket_name, file_name):
        self.job_uri = f'https://{bucket_name}.s3.us-west-1.amazonaws.com/{file_name}'
        self.job_name = 'check_your_tone_{:%Y%m%d_%H%M%S}'.format(datetime.utcnow())
        self.load_tokenizer()
        self.load_rnn()
        self.load_cnn()

    def load_tokenizer(self):
        self.tokenizer = pickle.load(open("tokenizer/sentence_tokenizer_fitted.sav", 'rb'))

    def tokenize_transcript(self):
        assert self.transcript
        # Convert text to tokenized vector
        X_processed = pre_process_sentence([self.transcript.lower()])
        X_tokenized = self.tokenizer.texts_to_sequences(X_processed)
        X_padded = pad_sequences(X_tokenized, padding='post', maxlen=100)
        self.transcript_tokenized = np.array(X_padded, dtype=np.float32)

    def get_transcript(self):
        """
        Initialize AWS client and get transcription from uploaded audio file. First, the remote task is initialized.
        Once complete, the text and metadata can be downloaded in JSON format.
        :return: Text transcript of audio file.
        """
        # Create client
        transcribe = boto3.client('transcribe',
                                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                  region_name='us-west-1')

        # Run job
        transcribe.start_transcription_job(TranscriptionJobName=self.job_name,
                                           Media={'MediaFileUri': self.job_uri},
                                           MediaFormat='wav',
                                           LanguageCode='en-US')

        # Check job status
        counter = 0
        while True:
            counter += 1
            status = transcribe.get_transcription_job(TranscriptionJobName=self.job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] == "FAILED":
                print("Transcription failed.")
                break

            elif status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                data = json.loads(response.read())
                self.transcript = data['results']['transcripts'][0]['transcript']
                self.tokenize_transcript()
                print(self.transcript)
                break

            else:
                if counter % 10 == 0:
                    print("Transcript is not ready...")
                time.sleep(1)

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
        # Allocate memory for model input tensors
        self.cnn_model.allocate_tensors()

    def predict_rnn(self):
        """
        From the transcript returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
        """

        input_X = self.transcript_tokenized

        ## RNN predictions
        # Set input and output tensors
        input_details = self.rnn_model.get_input_details()
        output_details = self.rnn_model.get_output_details()
        # Model prediction
        self.rnn_model.set_tensor(input_details[0]['index'], input_X)
        self.rnn_model.invoke()
        self.rnn_prediction = self.rnn_model.get_tensor(output_details[0]['index'])
        return self.rnn_prediction[0][0]  # Between 0 and 1

    def predict_cnn(self):
        """
        From the transcript returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
        """

        input_X = self.transcript_tokenized

        ## CNN predictions
        # Set input and output tensors
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
