#!/bin/bash

#################################################################################
# ============================================================================= #
# |-------------------------------CHECK YOUR TONE-----------------------------| #
# |	       A Speech-to-Text Sentiment Analyzer for Raspberry Pi	      | #
# |									      | #
# |	                Raspberry Pi Configuration Script		      | #
# |									      | #
# |         10.09.2020 - Started install from NOOBS v3.5 (09.15.2020)         | #
# |---------------------------------------------------------------------------| #
# ============================================================================= #
#################################################################################


# To execture this setup script, run the following from a terminal window in the
# same directory as this file: bash setup_py.sh
#
# Make sure you have configured your internet connection before running!

#################################################################################
# Create project directory
#################################################################################
mkdir /home/pi/Projects/Check-Your-Tone


#################################################################################
# Install Git Software for Version Control
#################################################################################

#Install
sudo apt update
sudo apt install -y git

#Configure
git config --global user.name "your.username"
git config --global user.email "your.email@mail.com"

#Check configuration
git config --list

#Clone project repository to local storage
git clone https://github.com/ericvc/Check-Your-Tone /home/pi/Projects/Check-Your-Tone


#################################################################################
# Install TensorFLow for Python (special instructions)
# Source: https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html
#################################################################################

sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev cython
sudo pip3 install pybind11
sudo pip3 install h5py
sudo pip3 install --upgrade setuptools
pip install gdown
# Copy/paste gdown binary
sudo cp /home/pi/.local/bin/gdown /usr/local/bin/gdown
# Download the wheel file from Google drive directory (150 MB)
gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
conda upgrade wrapt
# Install TensorFlow from downloaded wheel file
sudo -H pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl --ignore-installed


#################################################################################
# Install Python Libraries
#################################################################################

#Install RPi.GPIO
sudo apt-get -y install RPI.GPIO

#Install all over libraries with pip
pip3 install -r /home/pi/Projects/Check-Your-Tone/requirements.txt


#################################################################################
# Download, Install, and Configure Amazon Web Services CLI
#################################################################################

pip3 install awscli --upgrade --user

#Set AWS CLI executable as PATH variable
export PATH=/home/pi/.local/bin:$PATH

#Confirm installation
aws --version


#################################################################################
# Install Shutdown Button Script
#################################################################################

#Move shutdown button script
sudo mv /home/pi/Projects/Check-Your-Tone/py/shutdown_button.py /usr/local/bin/
sudo chmod +x /usr/local/bin/shutdown_button.py

#Move shutdown listener script to startup directory
sudo mv listen-for-shutdown.sh /etc/init.d/
sudo chmod +x /etc/init.d/listen-for-shutdown.sh

#Register script to run on boot
sudo update-rc.d listen-for-shutdown.sh defaults

#Initialize script
sudo /etc/init.d/listen-for-shutdown.sh start

