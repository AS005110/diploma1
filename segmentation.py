import cv2
import numpy as np

#поиск символов и их выделение

def bin_img(img):
    thresh_val = 101
    ret, binarized = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return binarized


def erode(img):
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=1)


def select_usefull(img):
    connectivity = 8
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    els = np.zeros((n_labels,))
    for n_lab in range(n_labels):
        els[n_lab] = len(np.where(labels == n_lab)[1])

    for noise in np.argsort(els)[:-10]:
        labels[labels == noise] = 0

    labels[labels != 0] = 255
    filtered = labels.astype('uint8')

    return filtered


def dilate(filtered):
    kernel_size = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    dilation = cv2.dilate(filtered, kernel, iterations=1)
    return dilation


def find_n_draw_contours(original, preprocessed):
    im1 = cv2.merge([original, original, original])
    contours, hierarchy = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im2 = im1.copy()
    indx = -1
    for contour in contours:
        # Bounding rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        im2 = cv2.drawContours(im2, [box], indx, (100, 300, 255), 1)
        indx -= 1
    return contours, im2
