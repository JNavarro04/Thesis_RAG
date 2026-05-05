import os
import datetime

logs = []


def debug_log(py_file_name, logs, text):
    momentary_datetime = "===\n" + py_file_name + str(datetime.datetime.now()) + "\n" + text + "\n"
    logs.insert(0, momentary_datetime)


def prepend_multiple_lines(file_name, list_of_lines):
    """Insert given list of strings as a new lines at the beginning of a file"""
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open given original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Iterate over the given list of strings and write them to dummy file as lines
        for line in list_of_lines:
            write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def run(list_of_lines):
    prepend_multiple_lines("logs.txt", list_of_lines)


if __name__ == '__main__':
    run()
