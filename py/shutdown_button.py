import RPi.GPIO as GPIO
import subprocess


#GPIO pin location (BCM)
SHUTDOWN_BUTTON = 22


GPIO.setmode(GPIO.BCM)
GPIO.setup(SHUTDOWN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.wait_for_edge(SHUTDOWN_BUTTON, GPIO.FALLING)


subprocess.call(["shutdown", "-h", "now"], shell=False)