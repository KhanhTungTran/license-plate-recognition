import cv2
import numpy as np
from numpy.lib.function_base import angle
from skimage import measure
from skimage.segmentation import clear_border
from imutils import perspective
import imutils
from data_utils import order_points, convert2Square, draw_labels_and_boxes
from detect import detectNumberPlate
from model import CNN_Model
from skimage.filters import threshold_local
import math

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

MIN_PIXEL_AREA = 40

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate()
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights('./weights/weight.h5')
        self.candidates = []
        self.preLpCnt = None


    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate


    def predict(self, image):
        # Input image or frame
        self.image = image

        for coordinate in self.extractLP():     # detect license plate by yolov3
            self.candidates = []

            x_min, y_min, width, height = coordinate
            LpRegion = self.image[y_min:y_min+height, x_min:x_min+width]

            # segmentation
            self.segmentation(LpRegion)

            # recognize characters
            self.recognizeChar()

            # format and display license plate
            license_plate = self.format()

            if len(license_plate) < 8:
                continue
            # draw labels
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        # cv2.imwrite('example.png', self.image)
        return self.image


    def check_four_corners(self, rec):
        topLeft = topRight = bottomLeft = bottomRight = 0
        w, h = rec.shape[:2]
        for corner in rec:
            if corner[0] < w/2 and corner[1] < h/2:
                topLeft += 1
            elif corner[0] > w/2 and corner[1] < h/2:
                topRight += 1
            elif corner[0] < w/2 and corner[1] > h/2:
                bottomLeft += 1
            else:
                bottomRight += 1
        return topLeft == topRight == bottomLeft == bottomRight == 1


    def clean_border(self, LpRegion):
        LpRegion = imutils.resize(LpRegion, width=400)
        lab = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        LpRegion = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # gray = clahe.apply(gray)
        blur1 = cv2.GaussianBlur(gray, (11,11), cv2.BORDER_CONSTANT)
        blur2 = cv2.GaussianBlur(gray, (25,25), cv2.BORDER_CONSTANT)
        difference = blur2 - blur1
        # _, difference = cv2.threshold(difference, 127, 255, 0)
        difference = clear_border(difference)
        difference = cv2.adaptiveThreshold(difference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 9)
        # cv2.imshow("gray", difference)
        # edged = cv2.Canny(gray, 50, 125)
        # cv2.imshow("edged", edged)
        cnts = cv2.findContours(difference.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:3]
        # print(cnts)
        lpCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True) # TODO: Playing around with this precision value
            if len(approx) == 4:
                lpCnt = approx
                self.preLpCnt = lpCnt
                break
        
        if lpCnt is not None and self.check_four_corners(self.preLpCnt):
            return LpRegion
        # cv2.drawContours(LpRegion, [self.preLpCnt], -1, (255, 255, 0), 3)
        # cv2.imshow("ROI", LpRegion)
        # cv2.waitKey(0)


    def segmentation(self, LpRegion):
        LpRegion = self.clean_border(LpRegion)
        # cv2.imshow("edge", edged)
        
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        # adaptive threshold
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = clear_border(thresh)
        # cv2.imwrite("step2_2.png", thresh)
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # try:
        #     lines = cv2.HoughLinesP(image=thresh,rho=1,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=200,maxLineGap=20)
        #     angle = 0
        #     num = 0
        #     thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        #     for line in lines:
        #         my_degree = math.degrees(math.atan2(line[0][3]-line[0][1], line[0][2]-line[0][0]))
        #         if -45 < my_degree < 45:
        #             angle += my_degree
        #             num += 1
        #         cv2.line(thresh, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0))
        #     angle /= num

        #     cv2.imshow("draw", thresh)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     # cv2.imwrite("draw.png", thresh)
        #     # Rotate image to deskew
        #     (h, w) = thresh.shape[:2]
        #     center = (w // 2, h // 2)
        #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
        #     thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # except:
        #     pass


        # edges = cv2.Canny(thresh,100,200)
        # thresh = cv2.medianBlur(thresh, 5)
        # cv2.imshow("thresh", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite("thresh.png", thresh)
        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

        # loop over the unique components
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue

            # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255
            # find contours from mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if h*w > MIN_PIXEL_AREA and 0.25 < aspectRatio < 1.0 and solidity > 0.2 and 0.35 < heightRatio < 2.0:
                    # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    # cv2.imwrite('./characters/' + str(y) + "_" + str(x) + ".png", cv2.resize(square_candidate, (56, 56), cv2.INTER_AREA))
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    # cv2.imshow("square_candidate", square_candidate)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    self.candidates.append((square_candidate, (y, x)))

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
