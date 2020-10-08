import re
from nltk.corpus import stopwords

"""
Functions for processing raw text. Removes or replaces special characters, empty spaces, etc.
"""

stop_words = stopwords.words("english")
def REPLACE_STOP_WORDS_NO_SPACE(x):
    # list comprehension to split input string into list of words, then evaluate each word
    words = [word for word in x.split() if word not in stop_words]
    # recombine the list of remaining words into a string
    words_no_stop = " ".join(words)
    return words_no_stop


def REPLACE_ELLIPSES_WITH_SPACE(x):
    return re.compile("\\.{2,}").sub(" ", x)


def REPLACE_CHARACTER_NO_SPACE(x):
    return re.compile("[\\.\\-;:!\'?,\"()\[\]\/]").sub("", x)


def REPLACE_BLANK_START_NO_SPACE(x):
    return re.compile("^\\s+").sub("", x)


def REPLACE_BLANK_END_NO_SPACE(x):
    return re.compile("\\s+$").sub("", x)


def REPLACE_BLANK_WITH_SPACE(x):
    return re.compile("\\s{2,}").sub(" ", x)


def REPLACE_FORMAT_NO_SPACE(x):
    return re.compile("&\\w").sub(" ", x)


def pre_process_sentence(sentences):
    sentences = [REPLACE_ELLIPSES_WITH_SPACE(line) for line in sentences]
    sentences = [REPLACE_CHARACTER_NO_SPACE(line) for line in sentences]
    sentences = [REPLACE_FORMAT_NO_SPACE(line) for line in sentences]
    sentences = [REPLACE_BLANK_START_NO_SPACE(line) for line in sentences]
    sentences = [REPLACE_BLANK_END_NO_SPACE(line) for line in sentences]
    sentences = [REPLACE_BLANK_WITH_SPACE(line) for line in sentences]
    sentences = [REPLACE_STOP_WORDS_NO_SPACE(line) for line in sentences]

    return sentences
