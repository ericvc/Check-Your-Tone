# /etc/init.d/shutdown_button.py

import RPi.GPIO as GPIO
import subprocess


#GPIO pin location
SHUTDOWN_BUTTON = 37


GPIO.setmode(GPIO.BOARD)
GPIO.setup(SHUTDOWN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.wait_for_edge(SHUTDOWN_BUTTON, GPIO.FALLING)


subprocess.call(["shutdown", "-h", "now"], shell=False)