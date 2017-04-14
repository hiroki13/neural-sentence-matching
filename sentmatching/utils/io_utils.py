import sys

PAD = '<padding>'
UNK = '<unk>'
SPACE = ' '
UNDER_BAR = '_'


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()
