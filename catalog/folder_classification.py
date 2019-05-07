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
from catalog.utils import folder_utils
import math

class folder_catalog:
    def __init__(self, logging, config, modeldir, classifier_filename,npy):
        self.img_path=  config.get('general', 'img_path')
        self.img_path_out=config.get('general', 'img_path_out')
        self.train_img=config.get('general', 'train_img')
        self.included_extensions = config.get('general', 'included_extensions').split(',')
        self.included_extensions_video = config.get('general', 'included_extensions_video').split(',')
        self.img_path_out=config.get('general', 'img_path_out')

        self.modeldir = modeldir
        self.classifier_filename = classifier_filename
        self.npy = npy
        self.config= config
        self.logger = logging


    def main(self):
        fu=folder_utils(self.logger,self.config)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, self.npy)

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                frame_interval_video = 10
                batch_size = 1000
                image_size = 182
                input_image_size = 160

                HumanNames = os.listdir(self.train_img)
                HumanNames.sort()

                self.logger.info('Loading feature extraction model')
                facenet.load_model(self.modeldir)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]


                classifier_filename_exp = os.path.expanduser(self.classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                c = 0
                category= ""
                timestamp=0
                self.logger.info('Start recursively path ' + self.img_path)
                for root, dirs, files in os.walk(self.img_path, topdown=False):
                    for filename in files:
                        file_path=os.path.join(root, filename)
                        extension=os.path.splitext(filename)[1][1:]
                        if extension in self.included_extensions :
                            self.logger.info("Start input:"+file_path)

                            prevTime = 0
                            # ret, frame = video_capture.read()
                            #try:
                            frame = cv2.imread(file_path.encode('utf-8'),0)
                            #except:
                            #    continue
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
                                self.logger.info('Face Detected: %d' % nrof_faces)

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
                                            self.logger.error('face is too close')
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
                                            #self.logger.info(predictions)
                                            best_class_indices = np.argmax(predictions, axis=1)
                                            # print(best_class_indices)
                                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                            #plot result idx under box
                                            text_x = bb[i][0]
                                            text_y = bb[i][3] + 20
                                            #print('Result Indices: ', best_class_indices[0])
                                            self.logger.info('Photo is: ' + HumanNames[best_class_indices[0]])
                                            self.logger.info(best_class_probabilities)
                                            people.append(HumanNames[best_class_indices[0]])

                                            j=0
                                            for p in HumanNames:
                                                if predictions[0,j] > 0.1:
                                                    self.logger.info(HumanNames[j] + ': ' + str(predictions[0,j]))
                                                j+=1
                                        except:
                                            self.logger.error('!!!!!!!!!! Error on '+file_path)

                                        #print(people)
                                    prev_category= category
                                    prev_ts= timestamp
                                    category, timestamp = fu.catalog(file_path, people, prev_category, prev_ts)
                                else:
                                    self.logger.error('!!!!!!!!!! Unable to align')
                                    timestamp = fu.catalog_woface(file_path, category, timestamp)

                        elif extension in self.included_extensions_video :

                            self.logger.info("Start video input:"+file_path)

                            video_capture = cv2.VideoCapture(file_path.encode('utf-8'))
                            no_frames= int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                            #fps = video_capture.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
                            #print(str(fps))

                            p_frame = math.ceil(no_frames /100)
                            c_frame= 0
                            c = 0
                            people = []
                            end = False
                            while end is not True:
                                self.logger.info('frame ' + str(c_frame) + ' of ' + str(no_frames))

                                try:
                                    #ret, frame = video_capture.read()
                                    video_capture.set(1,c_frame);
                                    ret, frame = video_capture.read()

                                    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    #cv2.imwrite('_frame_'+str(c_frame)+'.jpg',gray)

                                    curTime = time.time()+1    # calc fps
                                    timeF = frame_interval_video

                                    if (c % timeF == 0):
                                        find_results = []

                                        if frame.ndim == 2:
                                            frame = facenet.to_rgb(frame)
                                        frame = frame[:, :, 0:3]
                                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                                        nrof_faces = bounding_boxes.shape[0]
                                        self.logger.info('Face Detected: %d' % nrof_faces)
                                        if nrof_faces > 0:
                                            det = bounding_boxes[:, 0:4]
                                            img_size = np.asarray(frame.shape)[0:2]

                                            cropped = []
                                            scaled = []
                                            scaled_reshape = []
                                            bb = np.zeros((nrof_faces,4), dtype=np.int32)

                                            for i in range(nrof_faces):
                                                emb_array = np.zeros((1, embedding_size))

                                                bb[i][0] = det[i][0]
                                                bb[i][1] = det[i][1]
                                                bb[i][2] = det[i][2]
                                                bb[i][3] = det[i][3]

                                                # inner exception
                                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                                    self.logger.error('face is too close')
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
                                                    #print(predictions)
                                                    best_class_indices = np.argmax(predictions, axis=1)
                                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                                    # print("predictions")
                                                    #print(best_class_indices,' with accuracy ',best_class_probabilities)

                                                    # print(best_class_probabilities)
                                                    if best_class_probabilities>0.1:
                                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                                        #plot result idx under box
                                                        text_x = bb[i][0]
                                                        text_y = bb[i][3] + 20
                                                        self.logger.info('Photo is: ' + HumanNames[best_class_indices[0]])
                                                        self.logger.info(best_class_probabilities)
                                                        if HumanNames[best_class_indices[0]]!= 'otros' and HumanNames[best_class_indices[0]] not in people:
                                                            people.append(HumanNames[best_class_indices[0]])
                                                        j=0
                                                        for p in HumanNames:
                                                            if predictions[0,j] > 0.1:
                                                                self.logger.info(HumanNames[j] + ': ' + str(predictions[0,j]))
                                                            j+=1

                                                except:
                                                    self.logger.error('!!!!!!!!!! Error on '+file_path)
                                        else:
                                            self.logger.info('Alignment Failure')

                                    c_frame = c_frame + p_frame

                                    if len(people)>1:
                                        end= True
                                    elif len(people)>0 and (c_frame/no_frames > 0.5) :
                                        end= True
                                    elif c_frame/no_frames > 0.7 :
                                        end= True
                                    elif c_frame>=no_frames :
                                        end= True

                                except:
                                    end= True
                                    self.logger.info('Error reading file')

                            video_capture.release()
                            cv2.destroyAllWindows()
                            self.logger.info("People in photo:")
                            self.logger.info(people)

                            if len(people)>0:
                                prev_category= category
                                prev_ts= timestamp
                                category, timestamp = fu.catalog(file_path, people, prev_category, prev_ts)
                            else:
                                self.logger.error('!!!!!!!!!! Anybody known')
                                timestamp = fu.catalog_woface(file_path, category, timestamp)

            self.logger.info("End catalog")