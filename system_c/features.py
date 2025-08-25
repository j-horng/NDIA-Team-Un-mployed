import cv2
def orb(nfeatures=2000, fast=12): return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast)
