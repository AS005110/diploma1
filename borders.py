import cv2

#выделение номерных знаков

def draw_borders(img_gray, plate_points_1, plate_points_2):
    img = cv2.merge([img_gray, img_gray, img_gray])
    img_plate_1 = cv2.polylines(img, [plate_points_1], True, (400, 300, 200), 10)
    img_plate_2 = cv2.polylines(img, [plate_points_2], True, (150, 100, 200), 10)
    return img_plate_1, img_plate_2
