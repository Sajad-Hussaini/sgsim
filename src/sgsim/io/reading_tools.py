"""Low-level file reading utilities."""
import pathlib
import zipfile


def read_file_from_zip(filename: str, zip_file: str) -> list[str]:
    """
    Read content of a file from a zip archive.

    Parameters
    ----------
    filename : str
        Name of the file inside the zip archive.
    zip_file : str
        Path to the zip archive.

    Returns
    -------
    list[str]
        Lines of the file content.

    Raises
    ------
    FileNotFoundError
        If filename is not found in the zip archive.
    IOError
        If reading the zip archive fails.
    """
    try:
        return zipfile.Path(zip_file, filename).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} not found in {zip_file}")
    except Exception as e:
        raise IOError(f"Error reading zip: {e}")


def read_file(file: str) -> list[str]:
    """
    Read lines from a text file.

    Parameters
    ----------
    file : str
        Path to the file.

    Returns
    -------
    list[str]
        Lines of the file content.

    Raises
    ------
    IOError
        If reading the file fails.
    """
    try:
        return pathlib.Path(file).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        raise IOError(f"Error reading file: {e}")