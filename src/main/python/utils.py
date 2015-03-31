from difflib import SequenceMatcher

# http://stackoverflow.com/questions/17388213/python-string-similarity-with-probability
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()