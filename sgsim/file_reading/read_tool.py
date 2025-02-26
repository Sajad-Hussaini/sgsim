import zipfile
import pathlib

def read_file_from_zip(filename, zip_path):
    """
    Read the content of a file from a zip file.
    """
    try:
        return zipfile.Path(zip_path, filename).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f'{filename} not found in {zip_path}')
    except Exception as e:
        raise IOError(f'Error in reading zip: {str(e)}')

def read_file(file_path):
    """
    Read the lines of a file.
    """
    try:
        return pathlib.Path(file_path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        raise IOError(f'Error in reading file: {str(e)}')