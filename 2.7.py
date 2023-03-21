import os
import sys
import numpy as np


def if_character(input_char):
    if input_char.isalpha():
        return True
    elif input_char == ' ':
        return True
    else:
        return False


def get_file_characters(filename, language_characters):
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        for input_char in line:
            if if_character(input_char):
                language_characters.append(input_char)
    # print(len(language_characters))
    return language_characters


def get_language_vector(path, language):
    chars = 'abcdefghijklmnopqrstuvwxyz '
    language_files = []
    files = os.listdir(path)
    for filename in files:
        if filename[0] == language:
            language_files.append(path + filename)
    language_characters = []
    for language_filename in language_files:
        language_characters = get_file_characters(language_filename, language_characters)
    language_total_len = len(language_characters)
    language_vector = []
    # print(language_characters)
    for cha in chars:
        count_cha = language_characters.count(cha)
        language_vector.append((count_cha + 0.5) / (language_total_len + 27 * 0.5))

    language_vector = np.array(language_vector)
    return language_vector


def main():
    np.set_printoptions(suppress=True)
    sys.stdout.reconfigure(encoding='utf-8')
    chars = 'abcdefghijklmnopqrstuvwxyz '
    path = './languageID/'


    files = os.listdir(path)
    files_num = len(files)
    e_files = []
    j_files = []
    s_files = []

    for filename in files:
        if filename[0] == 'e':
            e_files.append(path + filename)
        if filename[0] == 'j':
            j_files.append(path + filename)
        if filename[0] == 's':
            s_files.append(path + filename)

    pye = (len(e_files) + 0.5) / (files_num + 3 * 0.5)
    pyj = (len(j_files) + 0.5) / (files_num + 3 * 0.5)
    pys = (len(s_files) + 0.5) / (files_num + 3 * 0.5)


    e_vector = get_language_vector(path, 'e')
    j_vector = get_language_vector(path, 'j')
    s_vector = get_language_vector(path, 's')

    e_vector_log = np.log(e_vector)
    j_vector_log = np.log(j_vector)
    s_vector_log = np.log(s_vector)

    languages = ['English','Japanese','Spanish']
    for language_name in languages:
        for file_idx in ['10.txt','19.txt']:
            filename = './languageID/'+language_name[0].lower()+file_idx
            print(filename)
            test_characters = []
            test_characters = get_file_characters(filename, test_characters)
            test_char_count = []
            for cha in chars:
                test_char_count.append(test_characters.count(cha))
            test_char_count = np.array(test_char_count)
            e_pred_log = np.dot(e_vector_log.T, test_char_count) + np.log(pye)
            j_pred_log = np.dot(j_vector_log.T, test_char_count) + np.log(pyj)
            s_pred_log = np.dot(s_vector_log.T, test_char_count) + np.log(pys)

            predict_language = languages[np.argmax(np.array([e_pred_log, j_pred_log, s_pred_log]))]
            # print(e_pred_log)
            # print(j_pred_log)
            # print(s_pred_log)
            print('The predict language is',predict_language)
            print('The true language is', language_name)
            print(predict_language == language_name)


    return


if __name__ == '__main__':
    main()
