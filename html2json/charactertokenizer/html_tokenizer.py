from .core import CharacterTokenizer
import pickle

class HTMLTokenizer:
    """
    Tokenizer for HTML tables
    the key idea here is to tokenize the html tables into characters and tags
    each opening tag is a token, each closing tag is a token and each character is a token

    CharacterTokenizer is used to encode and decode the tokens, this is an open source character level tokenizer.
    https://github.com/dariush-bahrami/character-tokenizer
    """
    def __init__(self, html_data):
        self.tokenizer = self.__build(html_data)

    def __build(self, html_data):
        """
        Builds the tokenizer
        :param html_data: a list of html tables in the form of bs4.BeautifulSoup objects
        :return: CharacterTokenizer
        """
        html_regular_tokens = set()
        html_special_tokens = set()
        for html in html_data:
            for tag in html.find_all(True):
                # adding the opening and closing tags to the special tokens
                html_special_tokens.add(f"<{tag.name}>")
                html_special_tokens.add(f"</{tag.name}>")
            for char in html.get_text():
                # adding all characters to the regular tokens
                html_regular_tokens.add(char)
        # building and returning the tokenizer
        return CharacterTokenizer(characters=list(html_regular_tokens), special_tokens=list(html_special_tokens))

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
