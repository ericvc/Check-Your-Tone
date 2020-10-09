import os
import RPi.GPIO as GPIO
import time
import numpy as np
from py.RunTranscriptionJob import RunTranscriptionJob


AWS_ACCESS_KEY_ID = keys["ACCESS"]
AWS_SECRET_ACCESS_KEY = keys["ACCESS_SECRET"]


# GPIO pin settings
PUSH_BUTTON = 10
INDICATOR_LED = 17
NEGATIVE_LED = 16
NEUTRAL_LED = 22
POSITIVE_LED = 32


# GPIO options
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering scheme
GPIO.setwarnings(False)  # Disable warnings
GPIO.setup(PUSH_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Push-button input (initial value is off)
GPIO.setup(INDICATOR_LED, GPIO.OUT, pull_up_down=GPIO.PUD_DOWN)  # Status indicator LED output
GPIO.setup(NEGATIVE_LED, GPIO.OUT, pull_up_down=GPIO.PUD_DOWN)  # LED output (negative)
GPIO.setup(NEUTRAL_LED, GPIO.OUT, pull_up_down=GPIO.PUD_DOWN)  # LED output (neutral)
GPIO.setup(POSITIVE_LED, GPIO.OUT, pull_up_down=GPIO.PUD_DOWN)  # LED output (positive)


# LED control function
def turn_led_on(LED: int, length: float=10.0):
    """
    :param LED: GPIO pin connected to a diode
    :param length: Time to keep on, in seconds.
    :return: Nothing is returned
    """
    GPIO.output(LED, True)
    time.sleep(length)
    GPIO.output(LED, False)
    time.sleep(1)

    # Blink two times to indicate that the program is resetting
    for cycle in range(2):
        GPIO.output(LED, True)
        time.sleep(0.25)
        GPIO.output(LED, False)



def record_audio(channel):
    """
    :param channel: GPIO pin connected to an LED.
    :return: Nothing is returned.
    """
    # Turn on recording indicator LED
    GPIO.output(channel, True)

    # Recording for 15 seconds, adding timestamp to the filename and sending file to S3
    file_name = "audio_recording_%s".format(np.round(time.time(), 3))
    bucket_name = "checkyourtoneproject"
    cmd = f'arecord /home/pi/Projects/CYT/{file_name}.wav -D sysdefault:CARD=1 -d 15 -r 48000;' \
          f'aws s3 cp /home/pi/Projects/CYT/{file_name}.wav s3://{bucket_name}'
    os.system(cmd)

    # Turn off recording indicator LED
    GPIO.output(channel, False)


def task_handler():
    """
    :return: Nothing is returned.
    """
    # Start time
    start_time = time.time()

    # Get audio recording
    record_audio(channel=INDICATOR_LED)

    # Get transcript from AWS
    transcript = RunTranscriptionJob(bucket_name=bucket_name, file_name=file_name)
    transcript.get_transcript()
    predicted_sentiment = np.round(transcript.ensemble_prediction(), 2)

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


# Detect when push button is pressed, run function when it is
GPIO.add_event_detect(PUSH_BUTTON, GPIO.RISING, callback=task_handler, bouncetime=100)


try:
    # Main program loop
    while True:
        turn_led_on(INDICATOR_LED, 2.0)

except KeyboardInterrupt:
    print("Check Your Tone! closed using keyboard exit command.")

except:
    print("An error has occurred.")

finally:
    GPIO.cleanup()  # Clean program exit
