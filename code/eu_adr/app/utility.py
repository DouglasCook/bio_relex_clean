import os
import datetime


def build_filepath(caller, f_path):
    """
    Build full filepath based on location of calling function and relative filepath
    This means that the script can be called from anywhere and will still be able to find other directories?
    """
    basepath = os.path.dirname(caller)
    return os.path.abspath(os.path.join(basepath, f_path))


def split_sentence(sent, start1, end1, start2, end2):
    """
    Put divs around the entities so they will be highlighted on page
    """
    return sent[:start1], sent[end1 + 1:start2], sent[end2 + 1:]


def time_stamped(fname, fmt='%Y-%m-%d-%H-%M-%S{fname}'):
    """
    Add a timestamp to start of filename
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
