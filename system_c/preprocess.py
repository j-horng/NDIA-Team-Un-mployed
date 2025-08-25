import cv2
def prep_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
