import equipment_config
import azure.cognitiveservices.speech as speechsdk


def init_ms_azure_stt(language):
    speech_key = equipment_config.get_ma_key1()
    service_region = equipment_config.get_ma_location()
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = equipment_config.get_ma_language(language)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    return speech_recognizer
