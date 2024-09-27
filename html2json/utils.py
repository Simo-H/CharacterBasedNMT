import json
import os
from bs4 import BeautifulSoup


def remove_attrs(soup):
    """
    Remove all attributes from all tags in the html data
    :param soup:
    :return: soup
    """
    for tag in soup.find_all(True):
        tag.attrs = {}
    return soup


def remove_whitespace(soup):
    """
    Remove leading and trailing whitespaces from all tags in the html data
    :param soup:
    :return: soup
    """
    for tag in soup.find_all(True):
        if tag.string is None:
            continue
        tag.string = tag.string.strip()
    return soup


def clean_html(html_str):
    """
    Clean the html data by removing attributes and whitespaces
    from tags, and newlines from between tags.
    return the cleaned html as a string
    :param html_str:
    :return: str
    """
    html_str = html_str.replace(">\n<", "><")
    html = BeautifulSoup(html_str, 'html.parser')
    html = remove_attrs(html)
    html = remove_whitespace(html)
    html_str = str(html)
    return html_str


class TokenizedJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that prepare the json object for tokenization. The encoder adds square brackets
    around each element in the json string that is part of the json language syntax. This is done to make
    the tokenization process easier.
    """
    def encode(self, obj):
        def custom_format(value):
            if isinstance(value, dict):
                items = [f'[{json.dumps(k)}][:]{custom_format(v)}' for k, v in value.items()]
                return f'[{{]{"[,]".join(items)}[}}]'
            elif isinstance(value, list):
                items = [custom_format(v) for v in value]
                return f'[[]{"[,]".join(items)}[]]'
            else:
                return json.dumps(value)

        return custom_format(obj)


def load_data(html_dir, json_dir, as_string=False, limit=None):
    """
    Load the html and json data from the directories
    :param html_dir: a path to the directory containing the html data
    :param json_dir: a path to the directory containing the json data
    :param as_string: a boolean flag to return the data as strings or as
    python objects (beautifulsoup and dict respectively). default is False.
    :param limit: the number of files to load. default is None, which loads all files
    :return: a tuple of lists containing the html and json data respectively
    """
    html_data = []
    json_data = []
    num_files = min(len(os.listdir(html_dir)), len(os.listdir(json_dir)))
    if limit is not None:
        num_files = min(num_files, limit)
    for i in range(num_files):
        with open(f'{html_dir}/{i}_table.html') as f:
            html = clean_html(f.read())
            if as_string:
                html_data.append(html)
            else:
                html_data.append(BeautifulSoup(html, 'html.parser'))

        with open(f'{json_dir}/{i}_metadata.json') as f:
            parsed_json = json.load(f)
            if as_string:
                json_data.append(json.dumps(parsed_json, cls=TokenizedJSONEncoder))
            else:
                json_data.append(parsed_json)
    return html_data, json_data
