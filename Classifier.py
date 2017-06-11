import cv2
import numpy as np
import scipy.misc
import csv
from Hist import *
from sklearn.neighbors import KNeighborsClassifier

def construct_classifier_data():
    make_test_samples()

def make_test_samples():
    training_data = []
    targets = []
    for i in range(1,15):
        image = scipy.misc.imread('Resources/training/images/img'+str(i))
        image = transformImage(image)
        mask = scipy.misc.imread('Resources/training/masks/img' + str(i)+'_mask')
        x_dim, y_dim,c = image.shape

        for y in range (6,y_dim-6,2):
            for x in range (6,x_dim-6,2):
                sample = image[x-6:x+6,y-6:y+6]
                sample_mask = mask[x-6:x+6,y-6:y+6]
                sample_attributes = extract_attributes(sample)
                training_data.append(sample_attributes)
                if(test_sample_mask(sample_mask)):
                    targets.append('1')
                else:
                    targets.append('0')
    save_training_to_csv(training_data,targets)

def extract_attributes(sample):
    attributes = []
    moments = cv2.moments(sample[0], False)
    huMoments = cv2.HuMoments(moments)
    for e in huMoments:
        attributes.append(str(e[0]))
    return attributes

def test_sample_mask(sample_mask):
    x_dim, y_dim = sample_mask.shape
    val=0
    for y in range(int(y_dim / 2) - 1, int(y_dim / 2) + 1):
        for x in range(int(x_dim/2)-1, int(x_dim/2)+1):
            val+= sample_mask[x,y]
    if(val>=6):
        return True
    else:
        return False

def save_training_to_csv(attributes,targets):
    with open('Resources/training/attributes.csv','w', newline='') as csvattributes:
        with open('Resources/training/targets.csv','w', newline='') as csvtargets:
            a_writer = csv.writer(csvattributes, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
            t_writer = csv.writer(csvtargets, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for a_row in attributes:
                a_writer.writerow(a_row)
            for t_row in targets:
                t_writer.writerow(t_row)

def read_training_data(mode):
    if(mode==0):
        attributes = []
        with open('Resources/training/attributes.csv','r') as csvattributes:
            a_reader = csv.reader(csvattributes, delimiter=' ', quotechar='|')
            for row in a_reader:
                tmp=[]
                for e in row:
                    tmp.append(float(e))
                attributes.append(tmp)
        return attributes
    else:
        targets = []
        with open('Resources/training/targets.csv', 'r') as csvattributes:
            a_reader = csv.reader(csvattributes, delimiter=' ', quotechar='|')
            for row in a_reader:
                for e in row:
                    targets.append(int(e))
        return targets


def make_samples(image):
    attributes=[]
    x_dim,y_dim,c= image.shape
    for y in range(6, y_dim - 6,2):
        for x in range(6, x_dim - 6,2):
            sample = image[x - 6:x + 6, y - 6:y + 6]
            sample_attributes = extract_attributes(sample)
            sample_attributes.append(str(x))
            sample_attributes.append(str(y))
            attributes.append(sample_attributes)
    save_attributes(attributes)

def save_attributes(attributes):
    with open('Results/attributes.csv','w', newline='') as csvattributes:
        a_writer = csv.writer(csvattributes, delimiter=' ',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for a_row in attributes:
            a_writer.writerow(a_row)

def read_attributes():
    attributes = []
    with open('Results/attributes.csv', 'r') as csvattributes:
        a_reader = csv.reader(csvattributes, delimiter=' ', quotechar='|')
        for row in a_reader:
            tmp = []
            for e in row:
                tmp.append(float(e))
            attributes.append(tmp)
    return attributes


def calculate_attributes(image):
    make_samples(image)
    return read_attributes()


# wydobycie cech obrazów treningowych z określoną przynależnością żyłka/przestrzeń /
# nauka klasyfikatora /
# testowanie klasyfikatora na zbiorze uczącym /
#
# Wariancja kolorów
