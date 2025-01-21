import os
import zipfile

def read_file_from_zip(filename, zip_path):
    """
    Read the content of a file from a zip file.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            with zip_file.open(filename, 'r') as file:
                return file.read().splitlines()
    except KeyError:
        raise FileNotFoundError(f'{filename} is not in the zip file.')
    except Exception as e:
        raise IOError(f'Error reading zip file: {str(e)}')

def read_file(file_path):
    """
    Read the lines of a file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except Exception as e:
        raise IOError(f'Error reading the record file: {str(e)}')
