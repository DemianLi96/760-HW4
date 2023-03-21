import os
import sys


def if_character(input_char):
    if input_char.isalpha():
        return True
    elif input_char == ' ':
        return True
    else:
        return False


def get_file_characters(filename):
    file = open('languageID/e0.txt', 'r')
    lines = file.readlines()
    file_characters = []
    for line in lines:
        for input_char in line:
            if if_character(input_char):
                file_characters.append(input_char)
    print(file_characters)
    print(len(file_characters))
    return file_characters


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    language_type = ['e', 'j', 's']
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

    print('p\u0302(y = e)=' + str((len(e_files) + 0.5) / (files_num + 3 * 0.5)))
    print('p\u0302(y = j)=' + str((len(j_files) + 0.5) / (files_num + 3 * 0.5)))
    print('p\u0302(y = s)=' + str((len(s_files) + 0.5) / (files_num + 3 * 0.5)))

    return


if __name__ == '__main__':
    main()
