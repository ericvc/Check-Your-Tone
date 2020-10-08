import os
import RPi.GPIO as GPIO
import time

# Setup GPIO options
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setwarnings(False)
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # pin 10 is push-button input (initial value is off)
GPIO.setup(18, GPIO.OUT)  # pin 18 is LED output

while True:

    if GPIO.input(10) = GPIO.HIGH:

        # Turn on recording indicator LED
        GPIO.output(18, GPIO.HIGH)

        # Recording for 20 seconds, adding timestamp to the filename and sending file to S3
        cmd = 'DATE_HREAD=$(date "+%s");arecord /home/pi/Projects/CYT/$DATE_HREAD.wav -D sysdefault:CARD=1 -d 20 -r 48000;' \
              'aws s3 cp /home/pi/Projects/CYT/$DATE_HREAD.wav s3://checkyourtoneproject'
        os.system(cmd)

        # Turn off recording indicator LED
        GPIO.output(18, GPIO.LOW)

    else:
        time.sleep(5e-3)  # sleep 5 ms
        # print ("Nothing detected")
