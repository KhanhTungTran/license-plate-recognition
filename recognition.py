import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils
from data_utils import order_points, convert2Square, draw_labels_and_boxes
from detect import detectNumberPlate
from model import CNN_Model
from skimage.filters import threshold_local
import pytesseract

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class E2E(object):
    def __init__(self):
        self.minAR = 1
        self.maxAR = 3
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate()
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights('./weights/weight.h5')
        self.candidates = []

    def extractLP(self, keep=5):
        # coordinates = self.detectLP.detect(self.image)
        # if len(coordinates) == 0:
        #     ValueError('No images detected')

        # for coordinate in coordinates:
        #     yield coordinate
        # perform a blackhat morphological operation that will allow
		# us to reveal dark regions (i.e., text) on light backgrounds
		# (i.e., the license plate itself)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

        # next, find regions in the image that are light
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # compute the Scharr gradient representation of the blackhat
		# image in the x-direction and then scale the result back to
		# the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
			dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")

        # blur the gradient representation, applying a closing
		# operation, and threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosions and dilations to clean up the
		# thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # take the bitwise AND between the threshold result and the
		# light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # find contours in the thresholded image and sort them by
		# their size in descending order, keeping only the largest
		# ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
		# return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, clearBorder=False):
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None
        box = None

        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR:
				# store the license plate contour and extract the
				# license plate from the grayscale image and then
				# threshold it
                lpCnt = c
                box = [x, y, x+w, y+h]
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # check to see if we should clear any foreground
				# pixels touching the border of the image
				# (which typically, but not always, indicates noise)
                if clearBorder:
                    roi = clear_border(roi)
				# display any debugging information and then break
				# from the loop early since we have found the license
				# plate region
                break
		# return a 2-tuple of the license plate ROI and the contour
		# associated with it
        return (roi, box)

    def predict(self, image):
        # Input image or frame
        self.image = image

        lpText = None
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        candidates = self.extractLP()
        (lp, box) = self.locate_license_plate(gray, candidates)

        if lp is not None:
			# OCR the license plate
            lpText = pytesseract.image_to_string(lp, config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            # draw labels
            self.image = draw_labels_and_boxes(self.image, lpText, box)
        
        print(lpText)
        # for coordinate in self.extractLP():     # detect license plate by yolov3
        #     self.candidates = []

        #     # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
        #     pts = order_points(coordinate)

        #     # crop number plate used by bird's eyes view transformation
        #     LpRegion = perspective.four_point_transform(self.image, pts)
        #     # cv2.imwrite('step1.png', LpRegion)
        #     # segmentation
        #     LpRegion = self.segmentation(LpRegion)

        #     # recognize characters
        #     # self.recognizeChar()
        #     predicted_result = pytesseract.image_to_string(LpRegion, lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        #     filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "") 
        #     print(filter_predicted_result)

        #     # # format and display license plate
        #     # license_plate = self.format()

        #     # draw labels
        #     self.image = draw_labels_and_boxes(self.image, filter_predicted_result, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        # apply thresh to extracted licences plate
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

        # adaptive threshold
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        cv2.imwrite("step2_1.png", thresh)
        return thresh
        # convert black pixel of digits to white pixel
        # thresh = cv2.bitwise_not(thresh)
        # cv2.imwrite("step2_2.png", thresh)
        # thresh = imutils.resize(thresh, width=400)
        # thresh = cv2.medianBlur(thresh, 5)

        # # connected components analysis
        # labels = measure.label(thresh, connectivity=2, background=0)

        # # loop over the unique components
        # for label in np.unique(labels):
        #     # if this is background label, ignore it
        #     if label == 0:
        #         continue

        #     # init mask to store the location of the character candidates
        #     mask = np.zeros(thresh.shape, dtype="uint8")
        #     mask[labels == label] = 255

        #     # find contours from mask
        #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #     if len(contours) > 0:
        #         contour = max(contours, key=cv2.contourArea)
        #         (x, y, w, h) = cv2.boundingRect(contour)

        #         # rule to determine characters
        #         aspectRatio = w / float(h)
        #         solidity = cv2.contourArea(contour) / float(w * h)
        #         heightRatio = h / float(LpRegion.shape[0])

        #         if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
        #             # extract characters
        #             candidate = np.array(mask[y:y + h, x:x + w])
        #             square_candidate = convert2Square(candidate)
        #             square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
        #             # cv2.imwrite('./characters/' + str(y) + "_" + str(x) + ".png", cv2.resize(square_candidate, (56, 56), cv2.INTER_AREA))
        #             square_candidate = square_candidate.reshape((28, 28, 1))
        #             self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))
    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate
