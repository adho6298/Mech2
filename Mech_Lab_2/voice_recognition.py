
# website: https://iotdesignpro.com/projects/speech-recognition-on-raspberry-pi-for-voice-controlled-home-automation
'''
microphone setup:
1. Connect a USB microphone to your Raspberry Pi.
2. command: alsamixer
F6, finding usb microphone, set it as default device.
F3: Playbacks, press m to turn off auto gain control
F4: Captures, set the volume to 70~80%, also make sure is on (press space)

For testing mocrophone, the command is:
arecord -D hw:3,0 -f cd -c 1 -d 5 -vv test.wav

after recording, play it back using:
aplay test.wav

After setting up the microphone, install the required libraries:
sudo apt update
sudo apt-get install espeak
sudo apt install flac

espeak is a text to speech library, which will be used to give voice feedback.



I installed SpeechRecognition and PyAudio in a virtual environment venv, type:
source venvs/speech/bin/activate
pip install SpeechRecognition
pip install PyAudio

To turn venv off, type:
deactivate


USB Mic
  ↓
ALSA
  ↓
PyAudio
  ↓
SpeechRecognition (audio object)(PCM --> FLAC)


Raspberry Pi
   |
   |  HTTPS POST
   |  (audio data)
   v
Google Speech Server

Then get result from google server. 


'''

import speech_recognition as sr
import subprocess
r = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Say something")
        audio = r.listen(source)
        print("Got it")
    return audio

def voice(audio):
    try:
        text = r.recognize_google(audio)
        print("You said:", text) # print out the text
        subprocess.run(["espeak", f"what you said is {text}"]) # play it back using espeak
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError:
        print("Google API error")
        return ""

if __name__ == "__main__":
    # while True:
    audio = listen()
    text = voice(audio)
