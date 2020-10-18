import os
import RPi.GPIO as GPIO
import time
import numpy as np
from py.RunTranscriptionJob import RunTranscriptionJob
import json


## GPIO pin settings
PUSH_BUTTON = 8
INDICATOR_LED = 36
NEGATIVE_LED = 11
NEUTRAL_LED = 16
POSITIVE_LED = 32


## GPIO options
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering scheme
GPIO.setwarnings(False)  # Disable warnings
GPIO.setup(PUSH_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Push-button input (initial value is on)
GPIO.setup(INDICATOR_LED, GPIO.OUT)  # Status indicator LED output
GPIO.setup(NEGATIVE_LED, GPIO.OUT)  # LED output (negative)
GPIO.setup(NEUTRAL_LED, GPIO.OUT)  # LED output (neutral)
GPIO.setup(POSITIVE_LED, GPIO.OUT)  # LED output (positive)


## AWS Authentication Settings
with open("/home/pi/Projects/Check-Your-Tone/amazon_tokens.json") as f:
    keys = json.load(f)

AWS_ACCESS_KEY_ID = keys["ACCESS"]
AWS_SECRET_ACCESS_KEY = keys["ACCESS_SECRET"]


## UX Settings
# Console window size
width = os.get_terminal_size().columns
# Indicator LED on
GPIO.output(INDICATOR_LED, True)


## Detect when tactile button is pressed, run function when it is
def event_listener():

    GPIO.add_event_detect(PUSH_BUTTON,
                          GPIO.FALLING,
                          callback=lambda x: task_handler(),
                          bouncetime=500)
    
    
## LED control function
def turn_led_on(LED: int, length: float=10.0, blinks: int=2):

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
    for cycle in range(blinks):
        GPIO.output(LED, True)
        time.sleep(0.25)
        GPIO.output(LED, False)


## Record audio and save to local storage
def record_audio(rlen: int=25):

    """
    :param channel: GPIO pin connected to an LED.
    :return: Nothing is returned.
    """
    
    # Disable event detection while function is running
    GPIO.remove_event_detect(PUSH_BUTTON)
    
    # Turn on recording indicator LED
    GPIO.output(NEGATIVE_LED, True)
    
    print("Recording started...\n")

    # Recording for 15 seconds, adding timestamp to the filename and sending file to S3
    frmt = "mp3"
    file_name = f"audio_recording_{np.round(time.time(), 3)}".replace(".","_")
    file_path = f"/home/pi/Projects/Check-Your-Tone/audio/{file_name}"
    bucket_name = "checkyourtoneproject"
    bit_rate = 96 # MP3 encoding bit rate
    cmd = f"arecord -f cd -t raw -d {rlen} -D plughw:2,0 | lame -r -b {bit_rate} - {file_path}.{frmt}"
    os.system(cmd)

    print("\nRecording ended...\n\nUploading audio file to AWS.\n")

    # Turn off recording indicator LED
    GPIO.output(NEGATIVE_LED, False)
    
    # Re-enable event detection
    event_listener()
    
    return bucket_name, file_name
    

## Function called by the event watcher when an edge event is detected
def task_handler():

    """
    :return: Nothing is returned.
    """

    # Start time
    start_time = time.time()
    
    # Get audio recording and upload to AWS project bucket"
    bucket_name, file_name = record_audio()
    file_path = f"/home/pi/Projects/Check-Your-Tone/audio/{file_name}.mp3"   

    # Get transcript from AWS - show light to indicate task is underway
    GPIO.output(NEUTRAL_LED, True)
    transcript = RunTranscriptionJob(bucket_name=bucket_name, 
                                    file_name=file_name, 
                                    az_key=AWS_ACCESS_KEY_ID, 
                                    az_secret=AWS_SECRET_ACCESS_KEY)
    transcript.run_parallel()  # Executes two functions (upload and model loading) in parallel to save time
    transcript.get_transcript()
    GPIO.output(NEUTRAL_LED, False)
    time.sleep(1)

    # Predict sentiment
    y_cnn, y_rnn = transcript.predict_ensemble()
    y_pred = np.mean([y_cnn, y_rnn])
    predicted_sentiment = np.round(y_pred, 3)

    # How long did the process take?
    task_id = transcript.job_name
    task_time = np.round(time.time() - start_time)

    # Report status
    if predicted_sentiment <= 0.33:

        print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
              f"NEGATIVE")
        print(f"\nCNN MODEL: {np.round(y_cnn,3)}, RNN MODEL: {np.round(y_rnn,3)}".center(width))
        turn_led_on(NEGATIVE_LED)

    elif 0.33 < predicted_sentiment <= 0.67:

        print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
              f"NEUTRAL")
        print(f"\nCNN MODEL: {np.round(y_cnn,3)}, RNN MODEL: {np.round(y_rnn,3)}".center(width))
        turn_led_on(NEUTRAL_LED)

    else:

        print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
              f"POSITIVE")
        print(f"\nCNN MODEL: {np.round(y_cnn,3)}, RNN MODEL: {np.round(y_rnn,3)}".center(width))
        turn_led_on(POSITIVE_LED)


## Initialize event listener
event_listener()


try:

    while True:

        continue

except KeyboardInterrupt:

    print("Check Your Tone! closed using keyboard exit command.")

except:

    print("An error has occurred.")

finally:

    # Indicator LED off
    GPIO.output(INDICATOR_LED, False)
    GPIO.cleanup()  # Clean program exit
