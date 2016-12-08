import os
import cv2


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:    # indicates that the left mouse button is pressed
        refPt = [(x, y)]
        cropping = True
    # elif event == cv2.EVENT_LBUTTONUP:
    elif event == cv2.EVENT_RBUTTONDOWN:   # indicates that the right mouse button is pressed
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


image_path = '/Users/pengfeiwang/Desktop/f/data/test_stg1'
output_path = '/Users/pengfeiwang/Desktop/output_test'
absolute_path = [os.path.join(image_path, i)
                 for i in os.listdir(image_path) if i[0] != '.']


loaction = dict()
for i in absolute_path:
    name = i.split('/')[-1].split('.')[0]
    image = cv2.imread(i)
    clone = image.copy()
    cv2.namedWindow("image")
    refPt = []
    cropping = False
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    if len(refPt) == 2:
        print refPt[0][1], refPt[1][1]
        print refPt[0][0], refPt[1][0]
        loaction[name] = [(refPt[0][1], refPt[1][1]),
                          (refPt[0][0], refPt[1][0])]
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.imwrite(os.path.join(output_path, i.split('/')[-1]), roi)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
