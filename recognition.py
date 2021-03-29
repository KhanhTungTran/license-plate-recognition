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

MIN_PIXEL_AREA = 60

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate()
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights('./weights/weight.h5')
        self.candidates = []

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

            # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # cv2.imshow("before Step1", self.image)
            # crop number plate used by bird's eyes view transformation
            LpRegion = perspective.four_point_transform(self.image, pts)
            # cv2.imwrite('step1.png', LpRegion)
            # cv2.imshow('step1', LpRegion)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # segmentation
            self.segmentation(LpRegion)

            # recognize characters
            self.recognizeChar()

            # format and display license plate
            license_plate = self.format()
            # predicted_result = pytesseract.image_to_string(LpRegion, lang ='eng', 
            # config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 

            # license_plate = "".join(predicted_result.split()).replace(":", "").replace("-", "") 

            if len(license_plate) < 8:
                continue
            # draw labels
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        # cv2.imwrite('example.png', self.image)
        return self.image


    def segmentation(self, LpRegion):
        # lab = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2LAB)

        # lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(1,1))

        # lab_planes[0] = clahe.apply(lab_planes[0])

        # lab = cv2.merge(lab_planes)

        # LpRegion = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        H = clahe.apply(H)
        S = clahe.apply(S)
        V = clahe.apply(V)
        hsv = cv2.merge((H, S, V))
        LpRegion = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("Lp", LpRegion)
        
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
