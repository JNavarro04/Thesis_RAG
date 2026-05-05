# change record
# 2023-11-24 Simplified demo version in English


# imported python modules
import os
import sys
import time
import datetime
import azure.cognitiveservices.speech as speech
from openai import OpenAI


# modules designed for JAIN project
import equipment_config
import speechToUnity
import remove_shelves
import usecases.uc027_legal_counter_openai
import experiment_settings
import statements


# nlp = spacy.load("en_core_web_lg")


def get_file_path(file_name=""):
    return os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), file_name)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_file_path("coach-1607434593693-e866839c8b17.json")


# logging
py_file_name = 'main: '
logs = []


def debug_log(text):
    momentary_datetime = "===\n" + py_file_name + str(datetime.datetime.now()) + "\n" + text + "\n"
    logs.insert(0, momentary_datetime)


def end_program():
    # print("Ending program.")
    # disconnect server
    # connect_photo_interrupt_sensor.send("!DISCONNECT")

    statement = "end of session"
    debug_log(statement)
    list_to_str = ' '.join(map(str, logs))

    log_file_path = equipment_config.get_project_path()
    filename = log_file_path + r"\logs.txt"
    file = open(filename, "a")
    file.write(list_to_str)
    file.write("\n")
    file.close()

    remove_shelves.run()
    # neo4j_manager.delete_path()
    sys.exit()


if __name__ == "__main__":
    # In case main has crashed in previous session, remove old shelve files
    # remove_shelves.run()
    language = experiment_settings.experiment_language()
    gender = experiment_settings.experiment_gender()
    text_strings = display_strings = {}
    if language == "nl":
        text_strings = statements.nl_text_strings
        display_strings = statements.nl_display_text_strings
    elif language == "en":
        text_strings = statements.en_text_strings
        display_strings = statements.en_display_text_strings
    elif language == "de":
        text_strings = statements.de_text_strings
        display_strings = statements.de_display_text_strings
    elif language == "tr":
        text_strings = statements.tr_text_strings
        display_strings = statements.tr_display_text_strings
    elif language == "es":
        text_strings = statements.es_text_strings
        display_strings = statements.es_display_text_strings
    elif language == "gr":
        text_strings = statements.gr_text_strings
        display_strings = statements.gr_display_text_strings
    elif language == "bn":
        text_strings = statements.bn_text_strings
        display_strings = statements.bn_display_text_strings
    elif language == "cn":
        text_strings = statements.cn_text_strings
        display_strings = statements.cn_display_text_strings

    # initiate Microsoft Azure speech to text:
    key1 = equipment_config.get_ma_key1()
    key2 = equipment_config.get_ma_key2()
    location = equipment_config.get_ma_location()
    endpoint = equipment_config.get_ma_endpoint()

    speech_key, service_region = key1, location
    speech_config = speech.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = equipment_config.get_ma_language(language)
    speech_recognizer = speech.SpeechRecognizer(speech_config=speech_config)

    # get openai key
    my_api_key = equipment_config.get_openai_key()
    my_organization = equipment_config.get_openai_organization()
    client = OpenAI(
        organization=my_organization,
        api_key=my_api_key,
    )
    # Start Unity server and wait until Unity is connected
    speechToUnity.start_unity_server()
    while speechToUnity.client == None:
        time.sleep(1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_file_path("coach-1607434593693-e866839c8b17.json")

    # usecases.uc027_legal_counter.dialogue_legal_advise()
    usecases.uc027_legal_counter_openai.dialogue_legal_advise()

    end_program()
