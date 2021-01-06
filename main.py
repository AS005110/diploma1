import cv2
import numpy as np
from matplotlib import pyplot as plt

import segmentation as seg
from borders import draw_borders
from recognition import crop_by_contours, char2img, classify
from recognition import crop_by_contours
from undisort import undisort
from utils import showInRow

plate_points_1 = np.array([[1028, 874], [1184, 890], [1184, 924], [1027, 906]], np.int32)
plate_points_2 = np.array([[293, 714], [422, 719], [421, 749], [291, 743]], np.int32)


# Borders around plates
def borders(img_gray):
    img_plates_lined1, img_plates_lined2 = draw_borders(img_gray, plate_points_1, plate_points_2)
    plt.imshow(img_plates_lined1)
    plt.waitforbuttonpress()


# Undisorting
def undisorting(img_gray):
    warped_img = undisort(img_gray, plate_points_1)
    plt.imshow(warped_img, cmap='gray')
    plt.waitforbuttonpress()
    return warped_img


# Segmentation
def segmentation(warped_img):
    img_gray_bw = seg.bin_img(warped_img)
    eroded = seg.erode(img_gray_bw)
    filtered = seg.select_usefull(eroded)
    dilated = seg.dilate(filtered)
    contours, drawed = seg.find_n_draw_contours(warped_img, dilated)
    plt.imshow(drawed)
    plt.waitforbuttonpress()
    return contours


# Recognition
def recognition(warped_img, contours):
    cropped_letters = crop_by_contours(warped_img, contours)

    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'X',
             'Y']
    chars_imgs = [char2img(char, 40, 30) for char in chars]

    # get hog
    hog = cv2.HOGDescriptor()
    # get descriptors of predefined char images
    descriptors_origin = np.array(
        [np.squeeze(hog.compute(np.array(img.resize((64, 128)))[..., 0])) for img in chars_imgs])
    descriptors_data = np.array([np.squeeze(hog.compute(cv2.resize(img, (64, 128)))) for img in cropped_letters])
    chars = np.array(chars)

    labels = [classify(query, descriptors_origin, chars) for query in descriptors_data]
    print(''.join(labels).lower())


def main():
    img_gray = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

    borders(img_gray)
    warped_img = undisorting(img_gray)
    contours = segmentation(warped_img)
    recognition(warped_img, contours)


if __name__ == '__main__':
    main()
