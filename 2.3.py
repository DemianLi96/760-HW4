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
    np.set_printoptions(precision=7)
    np.set_printoptions(suppress=True)
    sys.stdout.reconfigure(encoding='utf-8')
    chars = 'abcdefghijklmnopqrstuvwxyz '
    path = './languageID/'


    j_vector = get_language_vector(path, 'j')
    s_vector = get_language_vector(path, 's')
    print('\u03b8j is ', j_vector)
    print('\u03b8s is ', s_vector)
    return


if __name__ == '__main__':
    main()
