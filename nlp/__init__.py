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
    with open(path, 'wt') as w:
        w.write('[data]\n')
        for i in range(1, 4):
            url = f'https://raw.githubusercontent.com/tulane-cmps6730/project-reddit/main/data/{["train", "test", "validation"][i-1]}.csv'
            file = f'{nlp_path}{os.path.sep}{["train", "test", "validation"][i-1]}.csv'
            w.write(f'url{i} = {url}\n')
            w.write(f'file{i} = {file}\n')

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