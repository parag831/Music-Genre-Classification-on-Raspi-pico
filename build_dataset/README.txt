Data Collection Pipeline

The build_dataset.py contains a Hardware-in-the-Loop (HIL) audio data collection pipeline using a Raspberry Pi Pico.
Raw microphone samples are streamed over USB Serial, converted into WAV files , and uploaded to Google Drive.
The goal is to collect training data that closely matches real embedded deployment conditions , inorder to deal with mic bias when running inference 
on the hardware 
MIC (MAX9814A) was used as it contains Adaptive Gain Control or(AGC) 

Overview

Raspberry Pi Pico streams raw ADC audio samples over USB Serial
Python script reconstructs samples into PCM WAV files
User labels each recording
Files are automatically uploaded to Google Drive

Requirements
Software
Python 3.8 or newer
Python libraries:

    pip install pyserial soundfile numpy pydrive

Running the Pipeline
Connect the Raspberry Pi Pico via USB
Place client_secrets.json in the project directory (COM6 by default)

Run the script:
python build_dataset.py


Follow the prompts:
Wait for recording to complete
Enter y to save or n to discard
Provide a label (for example: classical, disco)
On first run, a browser window will open to authorize Google Drive access.

Configuration

Edit the following variables at the top of build_dataset.py.
Serial Port
PORT = 'COM6'

Windows: COM3, COM4, etc.
Linux: /dev/ttyACM0

after editing the code this python script also requires the client_secerets.json file. 
follow this page to create the .json file 
                https://stackoverflow.com/questions/65816603/how-to-generate-client-secret-json-for-google-api-with-offline-access
BAUD_RATE = 115200
