import RPi.GPIO as GPIO
import time
import numpy as np

## Dictionary with color:pin addresses
LED = {"RED":11, "YELLOW":16, "GREEN":32, "BLUE":36}


## GPIO options and setup
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering scheme
GPIO.setwarnings(False)  # Disable warnings

for color in LED:
    GPIO.setup(LED[color], GPIO.OUT)


## Main program
try:

    ## Function for turning an LED on and off
    def turn_led_on(pin: int, length: float=10.0):
        GPIO.output(pin, True)
        time.sleep(length)
        GPIO.output(pin, False)
        time.sleep(0.2)

    # Show each LED for 2 seconds in succession
    for color in LED:
        turn_led_on(LED[color], 1.25)
            
    # Each LED will flash 3 times in succesion
    for color in LED:
        for cycle in range(3):
            turn_led_on(LED[color], 0.25)
            
    # Cycle through colors quickly, 5 times
    for cycle in range(5):
        for color in LED:
            turn_led_on(LED[color], 0.08)
            
    # Cycle through LED colors by list
    colors = ["RED","YELLOW","GREEN","YELLOW","RED","GREEN","YELLOW","BLUE"]
    for color in colors:
        turn_led_on(LED[color], 0.25)

    # Cycle through LED colors by list and then in reverse
    colors = ["RED","YELLOW","GREEN","BLUE"]
    for color in colors:
        turn_led_on(LED[color], 0.35)
    time.sleep(0.25)
    for color in reversed(colors):
        turn_led_on(LED[color], 0.35)


    ## Random sequence of illumination
    def turn_on_random(pin_dict: dict, length: float=0.07, n_times: int=10):
        colors = list(pin_dict)
        for n in range(n_times):
            color = np.random.choice(colors)
            turn_led_on(pin_dict[color], length)

    # Cycle through LEDs randomly
    turn_on_random(LED, length=0.25, n_times=25)


    ## Turn all LEDs on
    def turn_all_led_on(pin_dict: dict, length: int=3):
        for color in pin_dict:
            GPIO.output(pin_dict[color], True)
        time.sleep(length)
        for color in pin_dict:
            GPIO.output(pin_dict[color], False)
        time.sleep(0.2)
            
    # Show all LEDs for 3 seconds
    pulse_length = [3,1,1,0.25,0.25,0.25]
    for length in pulse_length:
        turn_all_led_on(LED, length)

except KeyboardInterrupt:
    print("Test interrupted by keyboard exit command.")

except:
    print("An error has occurred.")

finally:
    GPIO.cleanup()  # Clean program exit


