from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from catalog.folder_classification import folder_catalog
import logging
import configparser
import os

config = configparser.RawConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)) + '/me.dev.properties'))
logging.basicConfig(filename="logs-organize-imgs.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'

print ("Folder classification Start")
obj=folder_catalog(logging,config, modeldir, classifier_filename,npy)
obj.main()
print('Finished classification')
sys.exit("All Done")
