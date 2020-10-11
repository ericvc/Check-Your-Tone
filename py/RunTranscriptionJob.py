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


## Suppress warnings from numpy
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)  


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
        file_format = file_name.split(".")[1]
        transcribe.start_transcription_job(TranscriptionJobName=self.job_name,
                                           Media={'MediaFileUri': self.job_uri},
                                           MediaFormat=file_format,
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

    def predict(self, model_name: str):
        """
        From the transcript returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
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
        self.cnn_prediction = self.predict(model_name="cnn")
        self.rnn_prediction = self.predict(model_name="rnn")
        return float(self.cnn_prediction), float(self.rnn_prediction)
