# Speaker Verification

## Description

This project is my final project at university, and it aims to build a speaker verification system with Python programming language. It uses Deep Learning and Digital Signal Processing techniques to verify the identity of a speaker. 

The system extracts unique features from a speaker's voice sample and compares them with previously stored features. Two methods are used to do this: the first is called Text-Dependent Speaker Verification, and the second is called Text-Independent Speaker Verification.

The speaker verification system can be used in various applications, including security systems, customer service, and personal voice assistants. It aims to provide a reliable and efficient way to verify a speaker's identity using just their voice.

## Installation
1. Clone the repository:

    ```shell
    git clone https://github.com/KyoGren/Speaker_Verification.git
    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```
## Usage

1. Run the main script at demonstration directory:

    ```shell
    python demonstration/src/main.py
    ```

2. The GUI will open. Click on the "Record Audio" button to record your voice or click on the "Upload file" button to upload your Audio file. The system will extract features from this audio and use it as the reference voice.

3. The keyword of Text-Dependent Speaker Verification is "Hey, Android".

3. To verify a speaker, click on the "Verify Speaker" button and record another voice sample. The system will compare the features of this sample with the reference voice and determine if it's from the same speaker.

4. The result of the verification will be displayed in the GUI.

Please note that the system needs a quiet environment to accurately extract features from the audio. Make sure to minimize background noise when recording voice samples.
