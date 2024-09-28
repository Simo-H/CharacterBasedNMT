from .core import CharacterTokenizer
import pickle


def get_keys(json_object):
    """
    Get all keys from a json in a recursive manner
    :param json_object: a valid json object (list or dict)
    :return: keys set
    """
    keys = set()
    if isinstance(json_object, list):
        for item in json_object:
            keys.update(get_keys(item))
    elif isinstance(json_object, dict):
        for key in json_object:
            keys.add(key)
            keys.update(get_keys(json_object[key]))
    return keys


def get_values(json_object):
    """
    Get all values from a json in a recursive manner
    :param dictionary:
    :return: values set
    """
    values = set()
    if isinstance(json_object, list):
        for item in json_object:
            values.update(get_values(item))
    elif isinstance(json_object, dict):
        for key, value in json_object.items():
            # values.add(value)
            values.update(get_values(json_object[key]))
    else:
        for value in json_object:
            values.add(value)
    return values


class JSONTokenizer:
    """
    A tokenizer for json data
    the key idea here is to tokenize the html tables into characters and json language tokens i.e. {, [
    but also the keys in the json

    CharacterTokenizer is used to encode and decode the tokens, this is an open source character level tokenizer.
    https://github.com/dariush-bahrami/character-tokenizer
    """
    def __init__(self, json_data):
        self.tokenizer = self.__build(json_data)

    def __build(self, json_data):
        """
        Build the tokenizer
        adding all characters and json tokens and keys to the tokenizer
        :param json_data:
        :return:
        """
        # building a tokenizer for the html data, each tag is a token and the characters inside the tags are also tokens
        # get a set of all tags in the html files
        json_regular_tokens = set()
        json_special_tokens = set()
        # adding the json structural tokens
        json_special_tokens.add("[{]")
        json_special_tokens.add("[}]")
        json_special_tokens.add("[:]")
        json_special_tokens.add("[,]")
        json_special_tokens.add("[[]")
        json_special_tokens.add("[]]")
        # add some tokens that need to be escaped
        json_regular_tokens.add("\"")
        json_regular_tokens.add("\\")
        # adding the json keys and values
        for json_file in json_data:
            # add all keys to the set
            for key in get_keys(json_file):
                json_special_tokens.add(f"[\"{key}\"]")
            # add all regular characters from the values to the set
            json_regular_tokens.update(get_values(json_file))
        return CharacterTokenizer(characters=list(json_regular_tokens), special_tokens=list(json_special_tokens))

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.tokenizer)

    def save(self, path):
        # save with pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # load with pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
