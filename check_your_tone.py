import os
import RPi.GPIO as GPIO
import time
import numpy as np
from py.RunTranscriptionJob import RunTranscriptionJob
import json
from py.upload_to_aws import upload_to_aws


## Suppress warnings from numpy
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 


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


## Detect when tactile button is pressed, run function when it is
def event_listener():
    GPIO.add_event_detect(PUSH_BUTTON,
                          GPIO.FALLING,
                          callback=lambda x: task_handler(),
                          bouncetime=500)
    
    
## LED control function
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


## Record audio and save to local storage
def record_audio():
    """
    :param channel: GPIO pin connected to an LED.
    :return: Nothing is returned.
    """
    
    # Disable event detection while function is running
    GPIO.remove_event_detect(PUSH_BUTTON)
    
    # Turn on recording indicator LED
    GPIO.output(NEUTRAL_LED, True)
    GPIO.output(NEGATIVE_LED, True)
    GPIO.output(POSITIVE_LED, True)
    
    print("Recording started...\n")

    # Recording for 15 seconds, adding timestamp to the filename and sending file to S3
    frmt = "mp3"
    file_name = f"audio_recording_{np.round(time.time(), 3)}".replace(".","_")
    file_path = f"/home/pi/Projects/Check-Your-Tone/audio/{file_name}"
    bucket_name = "checkyourtoneproject"
    rlen = 15  # length of recording
    bit_rate = 192 # MP3 encoding bit rate
    cmd = f"arecord -f cd -t raw -d {rlen} -D plughw:2,0 | lame -r -b {bit_rate} - {file_path}.{frmt}"
    os.system(cmd)

    print("\nRecording ended...\n\nUploading audio file to AWS.\n")

    # Turn off recording indicator LED
    GPIO.output(NEUTRAL_LED, False)
    GPIO.output(NEGATIVE_LED, False)
    GPIO.output(POSITIVE_LED, False)
    
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
    file_path = f"/home/pi/Projects/Check-Your-Tone/audio/{file_name}.wav"
    upload_to_aws(bucket_name=bucket_name, file_path=file_path, s3_file_name=file_name)

    # Get transcript from AWS
    transcript = RunTranscriptionJob(bucket_name=bucket_name, file_name=file_name)
    transcript.get_transcript()

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
        turn_led_on(NEGATIVE_LED)

    elif 0.33 < predicted_sentiment <= 0.67:
        print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
              f"NEUTRAL")
        turn_led_on(NEUTRAL_LED)

    else:
        print(f"{task_id} - {task_time} secs - Predicted sentiment score is {predicted_sentiment}: "
              f"POSITIVE")
        turn_led_on(POSITIVE_LED)


## Initialize event listener
event_listener()


try:
    # Main program loop
    while True:
        GPIO.output(INDICATOR_LED, True)
        time.sleep(7.0)
        GPIO.output(INDICATOR_LED, True)
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("Check Your Tone! closed using keyboard exit command.")

except:
    print("An error has occurred.")

finally:
    GPIO.cleanup()  # Clean program exit
