import cv2
import numpy as np
from skimage.filters import threshold_local
import imutils


def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def transformFourPoints(image, pts):
	
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


def show(title,what):
    
    cv2.imshow(title,what)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("img.jpg")
image = imutils.resize(image, height = 500)
orig = image.copy()
ratio = image.shape[0] / 500.0

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show('title',gray)

blurred = cv2.GaussianBlur(gray,(5,5),0)
show('blur',blurred)

edged = cv2.Canny(blurred,30,50)
show('edged',edged)

contours, hierarchy= cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)

    if len(approx) == 4:
        target = approx
        break

cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
show('draw', image)

warped = transformFourPoints(orig, target.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Original", imutils.resize(orig, height = 500))
cv2.imshow("Scanned", imutils.resize(warped, height = 500))
cv2.waitKey(0)
cv2.destroyAllWindows()
