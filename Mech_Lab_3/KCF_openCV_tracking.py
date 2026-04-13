import cv2, os, sys

print("cv2 version:", getattr(cv2, "__version__", "NO_VERSION_ATTR"))
print("cv2 path:", getattr(cv2, "__file__", "NO_FILE_ATTR"))
print("Has VideoCapture:", hasattr(cv2, "VideoCapture"))