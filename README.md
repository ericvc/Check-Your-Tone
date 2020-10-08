Check Your Tone
================

Introduction
------------

The **Check Your Tone** project is an attempt to create a near real-time speech-to-text sentinment analyzer. A mix of technologies underlie the project including (1) the Amazon Transcribe web service for automatic speech recognition, (2) Keras/TensorFlow for training a neural net text sentiment classifier, and (3) Raspberry Pi for audio data collection and software management.

Users will use the push button to make a short audio recording from a practice presentation or writing project. The recording will then be converted to text and subsequently analyzed to determine if the tone of the message skews negative, positive, or is neutral. Information about the analyzed recording will be presented to the user via the terminal and using indicator LEDs controlled by the Rasberry Pi.

All code for this project is written for Python 3.6+.

Future Directions
-----------------

-   A complete guide with step-by-step directions for building the project.
-   Downloadable disk image with all software files and dependencies pre-configured.

Feature Ideas
-------------

-   Indicator LED should blink while fetching ASR transcript. Might need to use `asyncio`.
-   Select length of recording using push button (1 tap, 2 taps, etc.).
-   Use word sentiment lexicon to identify the valence of individual words returned in the transcript.
