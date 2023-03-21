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


def main():
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
    e_characters = []
    for e_filename in e_files:
        e_characters = get_file_characters(e_filename,e_characters)
    e_total_len = len(e_characters)
    e_vector = []
    # print(e_characters)
    for cha in chars:
        count_cha =e_characters.count(cha)
        e_vector.append((count_cha+0.5)/(e_total_len+27*0.5))
    np.set_printoptions(precision=7)
    np.set_printoptions(suppress=True)
    e_vector = np.array(e_vector)
    print('\u03b8e is ',e_vector)

    return


if __name__ == '__main__':
    main()
