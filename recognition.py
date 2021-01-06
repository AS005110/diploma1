import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import numpy as np

#преобразование в символов в текст

def crop_by_contours(image, contours):
    symbols = {}
    for contour in contours:
        area = cv2.minAreaRect(contour)
        points = np.int0(cv2.boxPoints(area))
        XX = [point[0] for point in points]
        YY = [point[1] for point in points]
        x1 = min(XX)
        y1 = min(YY)
        x2 = max(XX)
        y2 = max(YY)
        symbols[x1] = image[y1: y2, x1: x2]

    return [symbols[x] for x in sorted(symbols.keys())]


def char2img(char, height, width):
    font_size = 65
    if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        font_size = 50
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    plate_font = ImageFont.truetype("plate_font.ttf", font_size)
    text_width, text_height = draw.textsize(char, font=plate_font)
    X = (width - text_width) / 2
    Y = (height - text_height) / 1.3
    draw.text((X, Y), char, font=plate_font)
    return image


def classify(query, originals, labels):
    dists = np.linalg.norm(originals - query, axis=1)
    return labels[np.argmin(dists)]

