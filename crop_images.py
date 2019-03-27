import os
import fnmatch
import cv2
import shutil

# generate cropped images from each face to use before data_preprocess.py


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('/usr/local/lib/python2.7/dist-packages/cv2/data/haarcascade_frontalface_alt.xml')

input_path = r"/face-recognition/crop_img"
output_path = r"/face-recognition/crop_img_out"
done_path = r"/face-recognition/crop_img_mov"

included_extensions = ['jpg','jpeg', 'png', 'gif', 'JPG', 'PNG', 'GIF', 'JPEG']
n=1
# keep update on every run
counter=1

for root, dirs, files in os.walk(input_path, topdown=False):
    for filename in files:
        file_path=os.path.join(root, filename)
        extension=os.path.splitext(filename)[1][1:]
        if extension in included_extensions :
            print("input:"+file_path)
            try:
                #  Loading the image to be tested
                test_image = cv2.imread(file_path)
                #plt.imshow(test_image, cmap='gray')
                test_image2 = test_image.copy()

                # Converting to grayscale as opencv expects detector takes in input gray scale images
                test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

                # Displaying grayscale image
                #plt.imshow(test_image_gray, cmap='gray')

                # Applying the haar classifier to detect faces

                faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);
                print('Faces found: ', len(faces_rects))

                for (x,y,w,h) in faces_rects:
                    cropped_temp = convertToRGB(test_image2)[y-80:y+h+80,x-80:x+w+80, :]
                    print("output:"+ output_path+'/image'+str(counter)+'.'+extension)
                    try:
                        cv2.imwrite(output_path+'/image'+str(counter)+'.'+extension,convertToRGB(cropped_temp))
                    except:
                        print("Error writing")
                    n=n+1
                #os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
                shutil.move(file_path, done_path+'/image'+str(counter)+'.'+extension)
            except:
                print("Error reading")
            counter=counter+1
