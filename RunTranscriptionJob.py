import boto3
import time
import urllib
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
from preprocessing import pre_process_sentence
import pickle
from keras.preprocessing.sequence import pad_sequences


class RunTranscriptionJob:

    def __init__(self, bucket_name, file_name):
        self.job_uri = f'https://s3.amazonaws.com/{bucket_name}{file_name}'
        self.job_name = 'check_your_tone_{:%Y%m%d_%H%M%S}'.format(datetime.utcnow())
        self.load_rnn()
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = pickle.load(open("tokenizer/sentence_tokenizer_fitted.sav", 'rb'))

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
                                  region_name='us-west-2')

        # Run job
        transcribe.start_transcription_job(TranscriptionJobName=self.job_name,
                                           Media={'MediaFileUri': self.job_uri},
                                           MediaFormat='mp3',
                                           LanguageCode='en-US')

        # Check job status
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=self.job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] == "FAILED":
                print("Transcription failed.")
                break

            elif status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                response = urllib.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
                data = json.loads(response.read())
                self.transcript = data['results']['transcripts'][0]['transcript']
                print(self.transcript)

            else:
                print("Transcript is not ready...")
                time.sleep(0.5)

    def load_rnn(self):
        """
        Loads RNN saved as a TensorFLow Lite model.
        """
        #Load model from file and allocate tensors
        self.model = tf.lite.Interpreter("TensorFlow Models/lstm_sentiment_classifier.tflite")
        # Allocate memory for model input tensors
        self.model.allocate_tensors()

    def predict_sentiment(self):
        """
        From the transcript returned by AWS, convert raw text to tokenized vector, and predict sentiment from the
        trained model.
        :return: A floating point value indicating whether the input skews negative (<=0.5) or positive (>0.5).
        """
        # Ensure transcript exists
        assert self.transcript
        # Convert text to tokenized vector
        X_processed = pre_process_sentence([self.transcript.lower()])
        X_tokenized = self.tokenizer.texts_to_sequences(X_processed)
        X_padded = pad_sequences(X_tokenized, padding='post', maxlen=100)
        input_X = np.array(X_padded, dtype=np.float32)
        # Set input and output tensors
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        # Model prediction
        self.model.set_tensor(input_details[0]['index'], input_X)
        self.model.invoke()
        self.prediction = self.model.get_tensor(output_details[0]['index'])
        return self.prediction[0][0]  # Between 0 and 1



