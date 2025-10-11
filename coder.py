import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import argparse
import json
import logging

from alive_progress import alive_bar

import traceback

# Methods
from Methods.MDCT_SVD_WEN import MDCT_SVD_WEN
from Methods.DIFFERENTIAL_SVD_XUE import DIFFERENTIAL_SVD_XUE 
from Methods.DIFFERENTIAL_SVD_XUE_MULTI import DIFFERENTIAL_SVD_XUE_MULTI
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2
from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3

# Tools
from lib.Analysis import Analysis
from lib.Attacks import Attacks
from lib.Tools import Tools
from lib.Secret import Secret

METHOD_CLASS_MAP = {
    "DIFFERENTIAL_SVD_XUE":                         DIFFERENTIAL_SVD_XUE,
    "DIFFERENTIAL_SVD_XUE_MULTI":                   DIFFERENTIAL_SVD_XUE_MULTI,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1":     DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2":     DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3":     DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1" : DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2" : DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2,
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3" : DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3,
    "MDCT_SVD_WEN_MULTI"                          : MDCT_SVD_WEN,
}

def run_test_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    tools = Tools()
    analysis = Analysis()
    attacks = Attacks()

    tests_to_reconstruct = []

    for test_to_reconstruct in config["performed_tests"]:
        method_class = METHOD_CLASS_MAP[test_to_reconstruct["method"]["name"]]
        method = method_class(test_to_reconstruct["method"]["params"])

        secret_arguments = {
            "seed" : test_to_reconstruct["secret"]["seed"],
            "type" : test_to_reconstruct["secret"]["type"],
        }

        secret = Secret(secret_arguments)

        carrier = tools.load_carrier(test_to_reconstruct["carrier"])

        tests_to_reconstruct.append({
            "method": method,
            "secret": secret,
            "carrier": carrier,
        })

    current_working_directory = os.getcwd()
    os.makedirs('Results', exist_ok=True)
    if not os.path.exists('Results'):
        os.makedirs('Results')

    tools.archive_previous_tests(results_root='Results', archive_folder='Archive')
    
    current_test_directory = os.path.join(current_working_directory, 'Results', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    os.makedirs(current_test_directory)

    attacks_to_performe = config["attacks"]

    with alive_bar(len(tests_to_reconstruct)) as bar:

        for test in tests_to_reconstruct:
            method = test["method"]
            secret = test["secret"]
            carrier = test["carrier"]

            method_directories = tools.prepare_method_directory(current_test_directory, method.name)
            method.set_method_directories(method_directories)
            method.set_carrier(carrier)
            method.set_secret(secret)

            stegobject = method.embedding()

            test_name = method.name + '_' + method.secret.name + '_' + carrier.name
            
            audio_to_save = {'name': test_name + '.wav', 'content': stegobject, 'fs': carrier.fs}
            tools.save_audio_file(method_directories['Stegobjects'], audio_to_save)

            attacks.perfome_attacks(method_directories['Stegobjects'], test_name, attacks_to_performe)

            stegobjects_paths, filenames = tools.find_files_with_string_in_name(method_directories['Stegobjects'], test_name, recursive=False)
            
            for stegobject_path, filename in zip(stegobjects_paths, filenames):
                
                loaded_stegobject = tools.load_audio_file(stegobject_path)
            
                extracted_data = method.extraction(loaded_stegobject, secret.size)
                
                if(method.secret.type == 'TXT'):
                    with open(os.path.join(method_directories['Extracted'], filename + '.txt'), 'w') as file:
                        file.write(extracted_data.get_bitvector_in_ascii())
                elif(method.secret.type == 'MAX_CAP'):

                    how_much = len(extracted_data) // 8
                    ext_sec = extracted_data[0:how_much * 8]

                    with open(os.path.join(method_directories['Extracted'], filename + '.txt'), 'w') as file:
                        file.write(ext_sec.get_bitvector_in_ascii())

                analysis.analyze(method, carrier, secret, loaded_stegobject, extracted_data)

            bar()

    analysis.save_methods_analysis_to_csv(current_test_directory)
    analysis.print_results()
    analysis.analyze_by_multiple_groups(["NAME", "TEST"])

def run_and_save_test():

    try:

        # TODO: Wyłączyć adaptację - i zobaczyć, jak to wpływa.
        # TODO: Czy zwiększenie poziomów, zwiększa odporność?
        # TODO: Czy zwiększenie liczby poziomów, zwiększa przeźroczystość?
        # TODO: Czy zwiększenie liczby poziomów, da się odtworzyć?

        # TEST = 'BITRATE_25_BPS'
        TEST = 'DEBUG'
        LOG_LEVEL = "DEBUG"

        if(LOG_LEVEL == 'INFO'):
            log_level = logging.INFO
        elif(LOG_LEVEL == 'DEBUG'):
            log_level = logging.DEBUG

        tools = Tools(log_level)

        logger = tools.get_logger()
        test_id = tools.get_test_id()
        current_test_directory = tools.get_current_test_directory()

                
        attacks = Attacks(logger=logger)
        analysis = Analysis(logger=logger)

        logger.info(f"Starting test with ID: {test_id}")

        if TEST == 'DEBUG':
            test_to_perform = {'NOISE' : False, 'HIGHPASS' : False, 'LOWPASS' : False, 'REQUANTIZE' : False, 'MP3' : False, 'JITTER' : False, 'RESAMPLE' : False, 'CLIPPED' : False}

            methods = [
                # 125 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 8,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 8, 'HOP_DURATION_MS' : 8}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 8, 'HOP_DURATION_MS' : 8, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 8, 'HOP_DURATION_MS' : 8, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 8, 'HOP_DURATION_MS' : 8, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 8, 'HOP_DURATION_MS' : 8, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # 100 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 10,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'ADAPTIVE' : False, 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'ADAPTIVE' : True , 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),                
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 8, 'HANKEL_ROWS' : 2}, logger),

                # # 62.5 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 32, 'HOP_DURATION_MS' : 16,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # # 50 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 20,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # # 40 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 50, 'HOP_DURATION_MS' : 25,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 25}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 25, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 25, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 25, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 25, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # # 25 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 40,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # # 12.5 BPS
                # MDCT_SVD_WEN(                                 {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 160, 'HOP_DURATION_MS' : 80,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 80}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 80, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 80, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 80, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 80, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
            ]
            
            # carriers = [tools.load_carrier('arctic_a0001')]
            carriers = tools.load_carriers()

        elif TEST == 'BITRATE_100_BPS':
            test_to_perform = {'NOISE' : False, 'HIGHPASS' : False, 'LOWPASS' : False, 'REQUANTIZE' : False, 'MP3' : False, 'JITTER' : False, 'RESAMPLE' : False, 'CLIPPED' : False}
            
            methods = [
                MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 10,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
            ]
            
            carriers = tools.load_carriers()

        elif TEST == 'BITRATE_80_BPS':
            test_to_perform = {'NOISE' : False, 'HIGHPASS' : False, 'LOWPASS' : False, 'REQUANTIZE' : False, 'MP3' : False, 'JITTER' : False, 'RESAMPLE' : False, 'CLIPPED' : False}
            
            methods = [
                MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 25, 'HOP_DURATION_MS' : 12.5,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 12.5, 'HOP_DURATION_MS' : 12.5}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 12.5, 'HOP_DURATION_MS' : 12.5, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 12.5, 'HOP_DURATION_MS' : 12.5, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 12.5, 'HOP_DURATION_MS' : 12.5, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 12.5, 'HOP_DURATION_MS' : 12.5, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
            ]
            
            carriers = tools.load_carriers()

        elif TEST == 'BITRATE_50_BPS':

            test_to_perform = {'NOISE' : False, 'HIGHPASS' : False, 'LOWPASS' : False, 'REQUANTIZE' : False, 'MP3' : False, 'JITTER' : False, 'RESAMPLE' : False, 'CLIPPED' : False}
            
            methods = [
                MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 20,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
            ]
            
            carriers = tools.load_carriers()

        elif TEST == 'BITRATE_25_BPS':
            test_to_perform = {'NOISE' : False, 'HIGHPASS' : False, 'LOWPASS' : False, 'REQUANTIZE' : False, 'MP3' : False, 'JITTER' : False, 'RESAMPLE' : False, 'CLIPPED' : False}
            
            methods = [
                MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 80, 'HOP_DURATION_MS' : 40,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
                DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
            ]

            carriers = tools.load_carriers()
        
        # methods = [

        #     # TEST
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.5, 'BETA': 0.5, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),

            

        #     # # 25 BPS

            
        #     # # 50 BPS
        #     # MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 20,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 20, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),

        #     # 62.5 BPS
        #     # MDCT_SVD_WEN(                                 {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 32, 'HOP_DURATION_MS' : 16,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.6, 'BETA': 0.6, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 16, 'HOP_DURATION_MS' : 16, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),

        #     # # 100 BPS
        #     DIFFERENTIAL_SVD_XUE(                         {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10}, logger),
        #     # MDCT_SVD_WEN(                                 {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20, 'HOP_DURATION_MS' : 10,'LEVELS' :  1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1(     {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(     {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3(     {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 4, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(  {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(  {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        #     # DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3(  {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 10, 'HOP_DURATION_MS' : 10, 'LEVELS' : 4, 'HANKEL_ROWS' : 2}, logger),

        #     # DIFFERENTIAL_SVD_XUE_MULTI(                   {'ALPHA': 0.4, 'BETA': 0.4, 'TYPE_OF_VAD' : 'rVAD', 'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40, 'HOP_DURATION_MS' : 40, 'LEVELS' : 1, 'HANKEL_ROWS' : 2}, logger),
        # ]

        methods_config = [{"name": method.__class__.__name__, "params": method.method_arguments} for method in methods]
        carrier_config = [carrier.to_metadata() for carrier in carriers]
        secrets_config = []
        performed_tests = []

        with alive_bar(len(methods) * len(carriers)) as bar:
            for method in methods:
                for carrier in carriers:
                    try:
                        
                        logger.info(f"Running test for method: {method.name}, carrier: {carrier.name}")
                        
                        method_directories = tools.prepare_method_directory(current_test_directory, method.name)
                        
                        method.set_method_directories(method_directories)
                        method.set_carrier(carrier)

                    except Exception as e:
                        logger.error(f"Error occurred while preparing method: {method.name}, carrier: {carrier.name}.")
                        continue
                    
                    try:
                        
                        try:
                            secret = Secret({'type' : 'MAX_CAP', 'size' : method.get_capacity(), 'save_directory' : method_directories['Secrets']}, logger=logger)
                        except Exception as e:
                            logger.error(f"Error creating secret for method: {method.name}, carrier: {carrier.name}. Error: {e}")
                            continue

                        method.set_secret(secret)

                        stegobject = method.embedding()
                        
                        secrets_config.append(secret.to_metadata())
                        
                        test_name = method.name + '_' + method.secret.name + '_' + carrier.name
                                                
                        tools.save_audio_file(method_directories['Stegobjects'], {'name' : test_name + '.wav', 'content' : stegobject, 'fs' : carrier.fs})
                        
                        attacks.perfome_attacks(method_directories['Stegobjects'], test_name, test_to_perform)
                        
                        stegobjects_paths, filenames = tools.find_files_with_string_in_name(method_directories['Stegobjects'], test_name, recursive=False)
                        
                        for stegobject_path, filename in zip(stegobjects_paths, filenames):
                            try:
                                loaded_stegobject = tools.load_audio_file(stegobject_path)
                                
                                extracted_data = method.extraction(loaded_stegobject)
                                
                                tools.save_extracted_secret(extracted_data, method_directories, filename)

                                analysis.analyze(method, carrier, secret, loaded_stegobject, extracted_data)
                            except Exception as e:
                                
                                logger.error(f"Error during extraction/analysis for {test_name}, file: {filename}")
                                logger.error(f"Error details: {e}")
                                logger.error(traceback.format_exc())
                                continue
                        
                        performed_tests.append({
                            "method": {
                                "name": method.__class__.__name__,
                                "params": method.method_arguments
                            },
                            "carrier": carrier.name,
                            "secret": {
                                "seed" : secret.seed,
                                "type": secret.type
                            },
                            "test_name": test_name
                        })
                    except Exception as e:
                        logger.error(f"Critical error in test for method: {method.name}, carrier: {carrier.name}")
                        logger.error(f"Error details: {e}")
                        logger.error(traceback.format_exc())
                        continue
                    
                    bar()
            
        
        print(f"TEST ID: {test_id}")
        analysis.save_methods_analysis_to_csv(current_test_directory)
        analysis.print_results()
        analysis.analyze_by_multiple_groups(["NAME", "TEST", "VAD", 'LEVELS', 'ALPHA', 'BETA', 'CAP. [bps]', 'ADAPTIVE'],     filters = None) #, 'BER [%]': lambda x: float(x) < 10
        analysis.save_grouped_to_csv(["NAME", "TEST", "VAD", 'LEVELS', 'ALPHA', 'BETA', 'CAP. [bps]', 'ADAPTIVE'], current_test_directory)

        config = {
            "methods": methods_config,
            "carriers": carrier_config,
            "secrets": secrets_config,
            "attacks": test_to_perform,
            "performed_tests": performed_tests
        }

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_config_{current_time}.json'
        with open(os.path.join(current_test_directory, filename), "w") as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        logger.error(f"An error occurred during the test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Steganography test runner")
    parser.add_argument("--config", type=str, help="Path to test_config.json")
    args = parser.parse_args()

    if args.config:
        run_test_from_config(args.config)
    else:
        run_and_save_test()