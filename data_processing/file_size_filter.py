import os
# FILTER Nº1 (FILE SIZE)
# CHECK THE SIZE OF ALL FILES IN THE files LIST, IF FILE_SIZE < 2.8MB WE REMOVE THAT FILE FROM THE files LIST
# WE END UP UP WITH THE UPDATED LIST OF FILES WE ARE INTERESTED
def get_file_size(file):
    return os.stat(file).st_size

def filter1(files, min_file_size=2800000):
    files = [file for file in files if get_file_size(file) >= min_file_size]
    return files
