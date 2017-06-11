import scipy.misc as misc
import matplotlib.pyplot as plt
from Filter import *
from Hist import *
from Classifier import *
from PIL import Image


if __name__ == '__main__':

    construct_classifier_data()
    kNN_attributes = read_training_data(0)
    targets = read_training_data(1)
    classifier = KNeighborsClassifier(n_neighbors=3)
    print("Classifier fiting")
    classifier.fit(kNN_attributes, targets)

    for img in range(1,21):
        image=scipy.misc.imread('Resources/training/images/img'+str(img))
        print(str(img)+' image read')
        image = transformImage(image)
        # image = cv2.medianBlur(image, 5)
        print(str(img) + " transformed")
        result = np.zeros_like(image)
        image_attributes = calculate_attributes(image)
        print(str(img) + ' attributes calculated')
        for sample in image_attributes:
            sample_attributes = sample[:-2]
            x = int(sample[-2])
            y = int(sample[-1])
            if(classifier.predict([sample_attributes])[0]):
                result[x-1:x+1,y-1:y+1]=255
        toSave = Image.fromarray(result)
        toSave.save('Results/'+str(img)+'_4_mask.jpg')

    # for x in range(1,21):
    #     image = scipy.misc.imread('Resources/training/images/img' + str(x))
    #     result = np.zeros_like(image)
    #     toSave = Image.fromarray(result)
    #     toSave.save('Results/'+str(x)+'mask.jpg')

    # image = scipy.misc.imread('Resources/test/images/1')

    # plt.subplot(1,2,1)
    # plt.imshow(result, cmap='gray')
    #
    # mask = scipy.misc.imread('Resources/test/masks/1')
    # mask = scipy.misc.imread('Resources/test/masks/img20_mask')
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='gray')
    # plt.show()


# Zapis wyników
# SK learn / orange

# Możliwe parametry :
# Momenty centralne, a nie Hu
# Liczba sąsiadów
# rysowanie tylko środkowego piksela, iterowanie po wszystkich
# wielkość sampla
#