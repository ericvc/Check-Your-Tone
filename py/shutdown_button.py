# /etc/init.d/shutdown_button.py

import RPi.GPIO as GPIO
import os


## GPIO pin location
SHUTDOWN_BUTTON = 37


## GPIO Setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SHUTDOWN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)


## Define listen event funtion
def event_listener():
    GPIO.add_event_detect(SHUTDOWN_BUTTON,
                          GPIO.FALLING,
                          callback=lambda x: os.system("sudo shutdown -h now"),
                          bouncetime=500)


event_listener()


try:

    while True:
        continue

except:

    print("An error has occurred. Shutdown button may not work.")


finally:

    GPIO.cleanup()  # Clean program exit
