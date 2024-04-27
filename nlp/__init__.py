# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """A Student"""
__email__ = 'student@example.com'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
    for i in range(1, 4):  # Iterate over url1, url2, url3
        data_url = config.get('data', f'url{i}', fallback=None)  # Fetch the 'url{i}' option
        if data_url is not None:
            data_file = config.get('data', f'file{i}')
            print('downloading from %s to %s' % (data_url, data_file))
            r = requests.get(data_url)
            with open(data_file, 'wt') as f:
                f.write(r.text)
        else:
            print(f"No URL found for 'url{i}' in configuration.")

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['HOME'] + os.path.sep + '.nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
clf_path = nlp_path + 'clf.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)