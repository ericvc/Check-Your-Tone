#!/bin/bash

#################################################################################
# ============================================================================= #
# |-------------------------------CHECK YOUR TONE-----------------------------| #
# |           A Speech-to-Text Sentiment Analyzer for Raspberry Pi            | #
# |                                                                           | #
# |                    Raspberry Pi Configuration Script                      | #
# |                                                                           | #
# |         10.09.2020 - Started install from NOOBS v3.5 (09.15.2020)         | #
# |---------------------------------------------------------------------------| #
# ============================================================================= #
#################################################################################


# To execture this setup script, run the following from a terminal window in the
# same directory as this file: bash setup_py.sh
#
# Make sure you have configured your internet connection before running!

#################################################################################
# Script Variables (edit for custom configuration)
#################################################################################

PROJDIR="/home/pi/Projects"  # Projects Directory
PROJNAME="Check-Your-Tone" #  Project Name
HOMEDIR="/home/pi/" #  Home Directory


#################################################################################
# Updates
#################################################################################
sudo apt-get update
sudo apt-get upgrade


#################################################################################
# Create project directory
#################################################################################
mkdir "${PROJDIR}/${PROJNAME}"


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
pip3 install -r "${PROJDIR}/${PROJNAME}/requirements.txt"

#Download datasets
python3 "${PROJDIR}/${PROJNAME}/config/dl_stopwords.py"


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
sudo cp "${PROJDIR}/${PROJNAME}/config/shutdown_button.py" /etc/init.d/
sudo chmod +x /etc/init.d/shutdown_button.py
sudo update-rc.d shutdown_button.py defaults


#################################################################################
# Configure Microphone and Speaker Settings (may require additional tweaking)
#################################################################################

#Move shutdown button script
sudo cp "${PROJDIR}/${PROJNAME}/config/.asoundrc" "${HOMEDIR}"
sudo chmod +x /home/pi/.asoundrc


#################################################################################
# Install FFmpeg
#################################################################################

# Run install script (may take some time)
sudo bash "${PROJDIR}/${PROJNAME}/config/ffmpeg_install.sh"


#################################################################################
# Install CHECK YOUR TONE!
#################################################################################

#Clone project repository to local storage
git clone https://github.com/ericvc/Check-Your-Tone "${PROJDIR}/${PROJNAME}"

#Configure program script to run at startup
sudo cp "${PROJDIR}/${PROJNAME}/check_your_tone.py" /etc/init.d/
sudo chmod +x /etc/init.d/check_your_tone.py
sudo update-rc.d check_your_tone.py defaults


#################################################################################
# Install remote desktop service (optional). Undo line comment to run.
#################################################################################

# Once installed, connect to this device using a remote desktop client

# Install
#sudo apt-get install -y xrdp

# Get local IP address
#hostname -I 


#################################################################################
# End of script.
#################################################################################

