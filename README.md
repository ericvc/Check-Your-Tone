Check Your Tone: A Speech-to-Text Sentiment Analyzer for Raspberry Pi
================

Introduction
------------

The **Check Your Tone** project is an attempt to create a near real-time speech-to-text sentinment analyzer. A mix of technologies underlie the project including (1) the Amazon Transcribe web service for automatic speech recognition, (2) Keras/TensorFlow for training a neural net text sentiment classifier, and (3) Raspberry Pi for audio data collection and software management.

Users will use the push button to initiate a short audio recording from, for example, a practice presentation or writing project. The recording will then be converted to text and subsequently analyzed to determine if the *tone* of the message skews negative, positive, or is neutral. Information about the analyzed recording will be presented to the user via the terminal and using indicator LEDs controlled by the Rasberry Pi.

All code for this project is written for Python 3.6+.

Installation (Raspberry Pi OS)
------------------------------

### Raspberry Pi Setup

Once the Raspberry Pi OS has been installed, download the file `setup_pi.sh` and run from a terminal window. This project is tested on a Raspberry Pi 3 Model B v1.2 (more detailed information can be found in `system.txt`).

    bash setup_pi.sh

-   Installs `shutdown_button.py` script, which assumes a shutdown button is connected to pin 22 (BCM)

Features
--------

### Text Analysis from CLI

Analyze the sentiment from text only using the command line interface and the `cyt.py` script:

    python cyt.py

    > Enter some text: This text sentiment analyzer will be useful when practicing for a presentation or drafting a writing project. It estimates the sentinment of example text using deep learning models that were trained on the IMDB movie reviews data set. See below for more information about how the models were created! I hope you enjoy using this project.

    > Predicted sentiment score is [0.44]: NEUTRAL

### Speech-to-Text Sentiment Analysis

Coming soon

Future Directions
-----------------

-   A complete guide with step-by-step directions for building the project.
-   Downloadable disk image with all software files and dependencies pre-configured.

Feature Ideas
-------------

-   Indicator LED should blink while fetching ASR transcript. Might need to use `asyncio`.
-   Select length of recording using push button (1 tap, 2 taps, etc.).
-   Use word sentiment lexicon to identify the valence of individual words returned in the transcript.
