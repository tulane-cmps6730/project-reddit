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
    w = open(path, 'wt')
    w.write('[data]\n')
    w.write('url1 = https://raw.githubusercontent.com/tulane-cmps6730/project-reddit/main/data/train.csv\n')
    w.write('file1 = %s%s%s\n' % (nlp_path, os.path.sep, 'train.csv'))
    w.write('url2 = https://raw.githubusercontent.com/tulane-cmps6730/project-reddit/main/data/test.csv\n')
    w.write('file2 = %s%s%s\n' % (nlp_path, os.path.sep, 'test.csv'))
    w.write('url3 = https://raw.githubusercontent.com/tulane-cmps6730/project-reddit/main/data/validation.csv\n')
    w.write('file3 = %s%s%s\n' % (nlp_path, os.path.sep, 'validation.csv'))
    w.close()

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