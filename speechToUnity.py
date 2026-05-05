# Original EvdB
# Modified RB line 119 server_thread.daemon = True to get exit = 0, see https://tinyurl.com/5y5pbwxv

import socket
import sys
import threading
import os
import time
from google.cloud import texttospeech
from mutagen.mp3 import MP3
import locale

import experiment_settings


# Get fullpip install path of file from its location relative to this script
def get_file_path(file_name=""):
    return os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), file_name)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_file_path("venv/coach-1607434593693-e866839c8b17.json")
audio_file = get_file_path("output.mp3")
client = None
is_speaking = False


def send_values_to_unity(is_running, duration):
    # Create a second TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the Unity Server on the same machine (localhost) and port 8052
    server_address = ("127.0.0.3", 8052)
    try:
        sock.connect(server_address)
        message = f'{is_running}, {duration}'.encode()
        sock.sendall(message)
    finally:
        sock.close()


def run_server():
    global client, is_speaking

    # Setup socket connection
    sock = socket.socket()
    sock.bind(('127.0.0.1', 5000))
    sock.listen(10)
    print('waiting connection...')

    # Client connects
    c, addr = client = sock.accept()

    # Listen for messages
    while True:
        try:
            bytes_received = c.recv(4096)
            if len(bytes_received) > 0:
                msg = str(bytes_received, 'utf8')
                # Print('message received: ' + msg)

                # When Unity connects, start manual text input for telling coach what to say
                if msg == 'Unity Connected':
                    print(msg)

                    # start user input
                    if __name__ == '__main__':
                        main_thread = threading.Thread(target=user_input)
                        main_thread.start()

                if msg == 'UnityStoppedSpeaking':
                    is_speaking = False
                    # print('Message UnityStoppedSpeaking received')

        except Exception as e:
            print(e)


# Sending messages to Unity
def _send(cmd, message):
    bytes_to_send = bytes(';'.join([cmd, message]), 'utf8')
    client[0].sendto(bytes_to_send, client[1])


def send_audio(path):
    _send('say', path)


def send_chat(speaker, transcript):
    _send('chat', ';'.join([speaker, transcript]))


def send_command(command):
    _send('do', command)


def send_media(path):
    _send('show', path)


def send_clearmedia():
    _send('show', '')


def say_something(text, language, gender):
    global is_speaking

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select voice type
    voice = get_voice(language, gender)

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    # The response's audio_content is binary.
    with open(audio_file, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

    # calculate length audio file
    audio = MP3("output.mp3")
    length_in_seconds = float(audio.info.length)
    is_running = True
    send_values_to_unity(is_running, length_in_seconds)
    print(f"Length of audio file: {length_in_seconds} seconds")

    # Send audio to Unity
    send_audio(audio_file)

    # Send chat transcript to Unity
    text.capitalize()
    send_chat('Avatar', text)

    # Assume we are speaking
    is_speaking = True


def user_input():
    language = experiment_settings.experiment_language()
    gender = experiment_settings.experiment_gender()
    while True:
        _input = input('What do you want me to do?')
        if _input.lower().startswith('do:'):
            send_command(_input[3:])
        elif _input.lower().startswith('show:'):
            if len(_input) > 5:
                send_media(get_file_path(_input[5:]))
            else:
                send_clearmedia()
        else:
            say_something(_input, language=language, gender=gender)
        while is_speaking:
            time.sleep(0.5)


def get_voice(requested_language, gender):
    if requested_language == 'nl' and gender == 'female':
        # standaard optie was nl-NL-Wavenet-A
        voice_out = texttospeech.VoiceSelectionParams(language_code="nl-NL", name="nl-NL-Chirp3-HD-Erinome",
                                                      ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'en' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="en-GB", name="en-GB-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'de' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="de-DE", name="de-DE-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'tr' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="tr-TR", name="tr-TR-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'es' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="es-ES", name="es-ES-Chirp3-HD-Erinome",
                                                      ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'gr' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="el-GR", name="el-GR-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'bn' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="bn-IN", name="bn-IN-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'cn' and gender == 'female':
        voice_out = texttospeech.VoiceSelectionParams(language_code="cmn-CN", name="cmn-CN-Chirp3-HD-Erinome",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
        return voice_out
    elif requested_language == 'nl' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="nl-NL", name="nl-NL-Chirp3-HD-Algenib",
                                                      ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'en' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="en-GB", name="en-GB-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'de' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="de-DE", name="de-DE-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'tr' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="tr-TR", name="tr-TR-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'es' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="es-ES", name="es-ES-Chirp3-HD-Algenib",
                                                      ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'gr' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="el-GR", name="el-GR-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'bn' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="bn-IN", name="bn-IN-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out
    elif requested_language == 'cn' and gender == 'male':
        voice_out = texttospeech.VoiceSelectionParams(language_code="cmn-CN", name="cmn-CN-Chirp3-HD-Algenib",
                                                  ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        return voice_out

def start_unity_server():
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()


if __name__ == '__main__':
    start_unity_server()
