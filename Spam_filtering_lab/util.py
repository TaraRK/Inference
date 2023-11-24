import os

def get_words_in_file(filename):
    """ Returns a list of all words in the file at filename. """
    with open(filename, 'r', encoding = "ISO-8859-1") as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of files in folder (including the path to the file) """
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

class Counter(dict):
    """
    Like a dict, but returns 0 if the key isn't found.

    This is modeled after the collections.Counter class, which is 
    only available in Python 2.7+. The full Counter class has many
    more features.
    """
    def __missing__(self, key):
        return 0

class DefaultDict(dict):
    """
    Like an ordinary dictionary, but returns the result of calling
    default_factory when the key is missing.

    For example, a counter (see above) could be implemented as either
    my_counter = Counter()
    my_counter = DefaultDict(lambda : 0)

    This is modeled after the collections.defaultdict class, which is 
    only available in Python 2.7+.
    """

    def __init__(self, default_factory):
        """
        default_factory is a function that takes no arguments and
        returns the default value
        """
        self._default_factory = default_factory

    def __missing__(self, key):
        return self._default_factory()

