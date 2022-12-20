import cv2 as cv
import os
import numpy as np
import math
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory

        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

    file_paths = os.listdir(root_path)
    return file_paths

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    images = []
    classes = []
    for idx, names in enumerate(train_names):
        path = root_path + '/' + names
        dir = os.listdir(path)
        for d in dir:
            true_path = root_path + '/' + names + '/' + d
            image = cv.imread(true_path)
            images.append(image)
            classes.append(idx)
    return images, classes


def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id

        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

    face_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    faces = []
    rectangles = []
    classes = []

    for i, img in enumerate(image_list):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(img_gray, 1.3, 6)

        if len(detected) < 1:
            continue

        for face in detected:
            x, y, w, h = face
            face_img = img_gray[y:y+h, x:x+w]

            faces.append(face_img)
            if image_classes_list is not None:
                classes.append(image_classes_list[i])

            rectangles.append([x, y, x+w, y+h])

    return faces, rectangles, classes


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id

        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory

        Returns
        -------
        list
            List containing all loaded gray test images
    '''

    test_imgs = []

    for filename in os.listdir(test_root_path):
        img = cv.imread(test_root_path + '/' + filename)
        test_imgs.append(img)

    return test_imgs

def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    classes = []

    for img in test_faces_gray:
        res, _ = recognizer.predict(img)
        classes.append(res)

    return classes


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''

    img_with_rect = []
    for result, img, rect in zip(predict_results, test_image_list, test_faces_rects):
        cv.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        text = train_names[result]
        cv.putText(img, text, (rect[0], rect[1]-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        img_with_rect.append(img)

    return img_with_rect


def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    for idx, img in enumerate(image_list):
        plt.subplot(2, 3, idx+1)
        plt.imshow(img)
        plt.axis("off")

    plt.show()

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, rec, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)

    combine_and_show_result(predicted_test_image_list)