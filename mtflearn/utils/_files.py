from pathlib import Path

def find_all_dm4_files(data_path):
    """
    Lists all .dm4 files in the specified directory.

    Parameters:
    data_path (str): The path to the directory to search for .dm4 files.

    Returns:
    list: A list of .dm4 file paths.
    """
    path = Path(data_path)
    dm4_files = list(path.rglob('*.dm4'))
    return [str(file) for file in dm4_files]
