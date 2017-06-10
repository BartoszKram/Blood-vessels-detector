import cv2
import numpy as np
import scipy.misc
import csv

def make_test_samples():
            for i in range(1,2):
                image = scipy.misc.imread('Resources/training/images/img'+str(i))
                mask = scipy.misc.imread('Resources/training/masks/img' + str(i)+'_mask')
                x_dim, y_dim, c = image.shape
                training_data = []
                targets = []
                for y in range (2,y_dim-3,5):
                    for x in range (2,x_dim-3,5):
                        sample = image[x-2:x+3,y-2:y+3]
                        sample_mask = mask[x-2:x+3,y-2:y+3]
                        sample_attributes = extract_attributes(sample)
                        training_data.append(sample_attributes)
                        if(test_sample_mask(sample_mask)):
                            targets.append('1')
                        else:
                            targets.append('0')
                save_training_to_csv(training_data,targets,i)


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
    for y in range(int(y_dim / 2) - 2, int(y_dim / 2) + 2):
        for x in range(int(x_dim/2)-2, int(x_dim/2)+2):
            val+= sample_mask[x,y]
    if(val>6):
        return True
    else:
        return False


# wydobycie cech obrazów treningowych z określoną przynależnością żyłka/przestrzeń
# nauka klasyfikatora
# testowanie klasyfikatora na zbiorze uczącym
# Wariancja kolorów

def save_training_to_csv(attributes,targets,i):
    with open('Resources/training/attributes'+str(i)+'.csv','w', newline='') as csvattributes:
        with open('Resources/training/targets'+str(i)+'.csv','w', newline='') as csvtargets:
            a_writer = csv.writer(csvattributes, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
            t_writer = csv.writer(csvtargets, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for a_row in attributes:
                a_writer.writerow(a_row)
            for t_row in targets:
                t_writer.writerow(t_row)