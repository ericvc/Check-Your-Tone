import tensorflow as tf
import numpy as np
from py.preprocessing import pre_process_sentence
from py.TextToneChecker import TextToneChecker
import pickle
from keras.preprocessing.sequence import pad_sequences
import sys
import os
import RPi.GPIO as GPIO


## GPIO pin settings
INDICATOR_LED = 36
NEGATIVE_LED = 11
NEUTRAL_LED = 16
POSITIVE_LED = 32


## GPIO options
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering scheme
GPIO.setwarnings(False)  # Disable warnings
GPIO.setup(INDICATOR_LED, GPIO.OUT)  # Status indicator LED output
GPIO.setup(NEGATIVE_LED, GPIO.OUT)  # LED output (negative)
GPIO.setup(NEUTRAL_LED, GPIO.OUT)  # LED output (neutral)
GPIO.setup(POSITIVE_LED, GPIO.OUT)  # LED output (positive)


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


## Console window width
width = os.get_terminal_size().columns


print("CHECK YOUR TONE! - Text Sentiment Analysis\nReturns score between 0 (negative) and 1 (positive)\n".center(width))

try:

    # Main program loop
    while True:

        text = sys.argv[1]
        tc = TextToneChecker(text)
        p_cnn, p_rnn = tc.predict_ensemble()
        y_pred = np.mean([p_cnn, p_rnn])
        predicted_sentiment = np.round(y_pred, 2)

        if predicted_sentiment <= 0.33:

            print(f"\nPredicted sentiment score is {predicted_sentiment}: NEGATIVE\n".center(width))
            print(f"\nCNN MODEL: {np.round(p_cnn,3)}, RNN MODEL: {np.round(p_rnn,3)}".center(width))
            turn_led_on(NEGATIVE_LED)

        elif 0.33 < predicted_sentiment <= 0.67:

            print(f"\nPredicted sentiment score is {predicted_sentiment}: NEUTRAL\n".center(width))
            print(f"\nCNN MODEL: {np.round(p_cnn,3)}, RNN MODEL: {np.round(p_rnn,3)}".center(width))
            turn_led_on(NEUTRAL_LED)

        else:

            print(f"\nPredicted sentiment score is {predicted_sentiment}: POSITIVE\n".center(width))
            print(f"\nCNN MODEL: {np.round(p_cnn,3)}, RNN MODEL: {np.round(p_rnn,3)}".center(width))
            turn_led_on(POSITIVE_LED)


except KeyboardInterrupt:

    print("Check Your Tone! closed using keyboard exit command.")

except:

    print("An unknown error has occurred.")

finally:
    
    # Indicator LED off
    GPIO.output(INDICATOR_LED, False)
    GPIO.cleanup()  # Clean program exit
    sys.exit(0)