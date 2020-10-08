import os
import RPi.GPIO as GPIO
import time
import numpy as np
from RunTranscriptionJob import RunTranscriptionJob


# !!! REQUIRES AMAZON CLI !!!

## Load API keys
with open("amazon_tokens.json") as f:
    keys = json.load(f)


AWS_ACCESS_KEY_ID = keys["ACCESS"]
AWS_SECRET_ACCESS_KEY = keys["ACCESS_SECRET"]


# GPIO Pin settings
PUSH_BUTTON = 10
INDICATOR_LED = 11
NEGATIVE_LED = 13
NEUTRAL_LED = 15
POSITIVE_LED = 16


# GPIO options
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering scheme
GPIO.setwarnings(False)
GPIO.setup(PUSH_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Push-button input (initial value is off)
GPIO.setup(INDICATOR_LED, GPIO.OUT)  # Status indicator LED output
GPIO.setup(NEGATIVE_LED, GPIO.OUT)  # LED output (negative)
GPIO.setup(NEUTRAL_LED, GPIO.OUT)  # LED output (neutral)
GPIO.setup(POSITIVE_LED, GPIO.OUT)  # LED output (positive)


# LED control function
def turn_led_on(LED: int, length: float=10.0):
    """
    :param LED: GPIO pin connected to a diode
    :param length: Time to keep on, in seconds.
    :return: Nothing is returned
    """
    GPIO.output(LED, GPIO.HIGH)
    time.sleep(length)
    GPIO.output(LED, GPIO.LOW)


while True:

    # Indicator light will blink slowly when program is idle
    while GPIO.input(PUSH_BUTTON) == GPIO.LOW:
        time.sleep(1)
        turn_led_on(INDICATOR_LED, 2.0)

    # Short pause between button action and recording
    time.sleep(0.25)

    if GPIO.input(PUSH_BUTTON) == GPIO.HIGH:

        # Start time
        start_time = time.time()

        # Turn on recording indicator LED
        GPIO.output(INDICATOR_LED, GPIO.HIGH)

        # Recording for 15 seconds, adding timestamp to the filename and sending file to S3
        file_name = "audio_recording_%s".format(np.round(time.time(), 3))
        bucket_name = "checkyourtoneproject"
        cmd = f'arecord /home/pi/Projects/CYT/{file_name}.wav -D sysdefault:CARD=1 -d 15 -r 48000;' \
              f'aws s3 cp /home/pi/Projects/CYT/{file_name}.wav s3://{bucket_name}'
        os.system(cmd)

        # Turn off recording indicator LED
        GPIO.output(INDICATOR_LED, GPIO.LOW)

        # Get transcript from AWS
        transcript = RunTranscriptionJob(bucket_name=bucket_name, file_name=file_name)
        transcript.get_transcript()
        predicted_sentiment = np.round(transcript.prediction(),2)

        # How long did the process take?
        task_id = transcript.job_name
        task_time = time.time() - start_time

        # Report status
        if predicted_sentiment <= 0.33:
            print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
                  f"NEGATIVE")
            turn_led_on(NEGATIVE_LED)

        elif 0.33 < predicted_sentiment <= 0.67:
            print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
                  f"NEUTRAL")
            turn_led_on(NEUTRAL_LED)

        else:
            print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
                  f"POSITIVE")
            turn_led_on(POSITIVE_LED)
