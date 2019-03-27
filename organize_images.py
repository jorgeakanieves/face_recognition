from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import sys
import shutil
import re
import configparser
config = configparser.RawConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)) + '/me.dev.properties'))
import logging

logging.basicConfig(filename="logs-organize-imgs.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

img_path=  config.get('general', 'img_path')
img_path_out=config.get('general', 'img_path_out')
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img=config.get('general', 'train_img')
included_extensions = config.get('general', 'included_extensions').split(',')
months = config.get('general', 'months').split(',')
cat_ar = config.get('general', 'cat_ar').split(',')
cat_ni = config.get('general', 'cat_ni').split(',')


def catalog(img_path, people, prev_category):
    m,y = extract_date(img_path)
    category = extract_category(people, prev_category)

    if category != None:
        logging.info('Extracted cat is ' + category)
        new_path=os.path.join(img_path_out, category, str(y), str(months[m-1]))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.move(img_path, os.path.join(new_path, os.path.basename(img_path)))
    else:
        logging.info('No extracted category')
    return category

def extract_date(img_path):
    date = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", img_path)
    date2 = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", img_path)
    if date != None and len(date) > 0 :
        date_arr = date.split('-')
        year = date_arr[2]
        month = date_arr[1]
        return month, year
    if date2 != None and len(date2) > 0 :
        date_arr = date2.split('-')
        year = date_arr[2]
        month = date_arr[1]
        return month, year
    else:
        created = os.path.getctime(img_path)
        year,month,day,hour,minute,second=time.localtime(created)[:-3]
        return month, year

def extract_category(people, prev_category):
    clause1 = clause2 = clause3 = clause4 = clause5 = clause6 = clause7 = 0
    for person in people:
        if eval(config.get('general', 'clause1')):
            clause1+=1
        if eval(config.get('general', 'clause7')):
            clause7+=1
        if eval(config.get('general', 'clause2')):
            clause2+=1
        if eval(config.get('general', 'clause3')):
            clause3+=1
        if eval(config.get('general', 'clause4')):
            clause4+=1
        if eval(config.get('general', 'clause5')):
            clause5+=1
        if eval(config.get('general', 'clause6')):
            clause6+=1

    if clause1 > 0 :
        return config.get('general', 'response1')
    if clause7 > 0 :
        return config.get('general', 'response7')
    if clause2 > 0 :
        return config.get('general', 'response2')
    if clause3 > 0 :
        return config.get('general', 'response3')
    if clause4 > 0 :
        return config.get('general', 'response4')
    if clause5 > 0 :
        return config.get('general', 'response5')
    if clause6 > 0 :
        return config.get('general', 'response6')

    return None


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        logging.info('Loading feature extraction model')
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        # video_capture = cv2.VideoCapture("akshay_mov.mp4")
        c = 0

        logging.info('Start recursively path ' + img_path)
        for root, dirs, files in os.walk(img_path, topdown=False):
            for filename in files:
                file_path=os.path.join(root, filename)
                extension=os.path.splitext(filename)[1][1:]
                if extension in included_extensions :
                    logging.info("Start input:"+file_path)

                    prevTime = 0
                    # ret, frame = video_capture.read()
                    frame = cv2.imread(file_path,0)

                    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                    curTime = time.time()+1    # calc fps
                    timeF = frame_interval

                    if (c % timeF == 0):
                        find_results = []

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        logging.info('Face Detected: %d' % nrof_faces)

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]

                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            bb = np.zeros((nrof_faces,4), dtype=np.int32)

                            people = []
                            for i in range(nrof_faces):
                                emb_array = np.zeros((1, embedding_size))

                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                    logging.error('face is too close')
                                    continue
                                try:
                                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                           interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    #logging.info(predictions)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    # print(best_class_indices)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                    #plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    #print('Result Indices: ', best_class_indices[0])
                                    logging.info('Photo is: ' + HumanNames[best_class_indices[0]])
                                    logging.info(best_class_probabilities)
                                    people.append(HumanNames[best_class_indices[0]])

                                    j=0
                                    for p in HumanNames:
                                        if predictions[0,j] > 0.1:
                                            logging.info(HumanNames[j] + ': ' + str(predictions[0,j]))
                                        j+=1
                                except:
                                    logging.error('!!!!!!!!!! Error on '+file_path)

                                #print(people)
                            prev_category= category
                            category = catalog(file_path, people, prev_category)
                        else:
                            logging.error('!!!!!!!!!! Unable to align')


                    #if cv2.waitKey(1000000) & 0xFF == ord('q'):
                    #    sys.exit("Thanks")
                    #cv2.destroyAllWindows()
        logging.info("End organize images")