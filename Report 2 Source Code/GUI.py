import tkinter as tk
from tkinter import filedialog

import sys
assert sys.version_info >= (3, 7)

import numpy as np
import cv2 as cv
from util_func import *

from imutils import paths
import matplotlib.pyplot as plt
import random
import re
import argparse

def preprocessing(img):
    
     #resize
    factor = 300 / img.shape[1] #scale factor = output width / original image width
    img_resize = cv.resize(img, None,fx=factor,fy=factor)
    
    #blur img using median Blur
    blur = cv.medianBlur(img_resize, 5)

    # CLAHE
    img_lab = cv.cvtColor(blur, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(img_lab)
    clahe = cv.createCLAHE(clipLimit = 10, tileGridSize = (4, 4))
    img_clahe = clahe.apply(l)
    img_merged = cv.merge((img_clahe, a, b))
    img_merged = cv.cvtColor(img_merged, cv.COLOR_Lab2BGR)

    gray = cv.cvtColor(img_merged, cv.COLOR_BGR2GRAY)

    #return resized img, blurred img, CLAHE img and grayscale img
    return img_resize, blur, img_merged, gray

def color_seg(img, img_blur, img_gray):
    
    #define hsv value range for red, blue, yellow and black

    red_low = (160, 115, 0)
    red_high = (180, 255, 255)

    red_low1 = (0, 115, 0)
    red_high1 = (10, 255, 255)

    blue_low = (100, 120, 0)
    blue_high = (120, 255, 255)

    yellow_low = (15, 120, 20)
    yellow_high = (30, 255, 255)

    black_low = (0, 20, 0)
    black_high = (180, 255, 45)
    
    #convert to hsv
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    
    #combine 2 red mask
    mask_red = cv.bitwise_or(cv.inRange(img_hsv, red_low, red_high), cv.inRange(img_hsv, red_low1, red_high1))
    
    #blue mask
    mask_blue = cv.inRange(img_hsv, blue_low, blue_high)
    
    #combine yellow and black mask
    mask_yellow = cv.bitwise_or(cv.inRange(img_hsv, yellow_low, yellow_high), cv.inRange(img_hsv, black_low, black_high))
    
    color_masks = []
    color_masks.append(mask_red) 
    color_masks.append(mask_blue) 
    color_masks.append(mask_yellow) 
    max_area = 0
    final_color_mask = []
    final_contour = []
    
    # for the 3 color masks, find the contour and compare the area
    for i in range(len(color_masks)):
        kernel = np.ones((3, 3), np.uint8)
        color_mask = color_masks[i]
        
        color_mask = cv.dilate(color_masks[i], kernel, iterations = 1)
        
        #get largest area of color mask
        area, contour_res = findContour(img, color_mask, img_gray)
        
        #get the color with largest contour area as the final contour
        if area > max_area:
            max_area = area
            final_color_mask = color_mask
            final_contour = contour_res

    #return final contour result (largest contour)
    return final_contour

# find contour
def findContour(img, mask, img_gray):
    
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    
    if len(contours) > 0:
        # Sort contours with regards to area in descending order and take the first
        contours = sorted(contours, key=cv.contourArea, reverse=True)[0]
        area = cv.contourArea(contours)

        # Create an empty mask
        contour_res = np.zeros_like(img_gray)

        img_copy = img.copy()
        cv.drawContours(contour_res, [contours], 0, (255), thickness=cv.FILLED)
    else:
        # No valid contours found, return an empty mask and area of 0
        contour_res = np.zeros_like(img_gray)
        area = 0

    return area, contour_res

def extractRegion(contour_res, img):
    extracted_region = cv.bitwise_and(img, img, mask=contour_res)
    
    #put text
    position = (17, 17)
    (text_width, text_height), _ = cv.getTextSize("Segmented Image", cv.FONT_HERSHEY_COMPLEX, 0.7, 2)
    background_rect_coords = ((position[0], position[1] - text_height), 
                          (position[0] + text_width, position[1]))
    cv.rectangle(extracted_region, background_rect_coords[0], background_rect_coords[1], (0, 0, 0), -1)
    cv.putText(extracted_region,"Segmented Image", (17, 17), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    return extracted_region

def bounding(mask, img):
    img_copy = img.copy()
    
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
     
    if len(contours) > 0:
        contours = sorted(contours, key =cv.contourArea, reverse=True)[0]
        x, y, w, h = cv.boundingRect(contours)
        
        #draw bounding boxes
        cv.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        # if no valid contours found, return 0
        x, y, w, h = [0,0,0,0]
        cv.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
     
    #put text
    position = (17, 17)
    (text_width, text_height), _ = cv.getTextSize("Bounded Image", cv.FONT_HERSHEY_COMPLEX, 0.7, 2)
    background_rect_coords = ((position[0], position[1] - text_height), 
                          (position[0] + text_width, position[1]))
    cv.rectangle(img_copy, background_rect_coords[0], background_rect_coords[1], (0, 0, 0), -1)
    cv.putText(img_copy, "Bounded Image", (17, 17), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_copy, x, y, w, h

##IOU
def convert_xywh_to_xyxy(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

def computeIOU(boxA, boxB):
    """The format of boxA and boxB is xyxy"""
    # compute the intersection area
    x_start = max(boxA[0], boxB[0])
    y_start = max(boxA[1], boxB[1])
    x_end = min(boxA[2], boxB[2])
    y_end = min(boxA[3], boxB[3])
    
    interArea = max(0, x_end - x_start + 1)* max(0, y_end - y_start + 1)
    
    # area of boxA and boxB
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    return interArea / (areaA + areaB - interArea)

def IOU(img, img_resize, img_path, x, y, w, h):
    
    img_copy = img_resize.copy()
    img_gt = img_resize.copy()
    factor = 300 / img.shape[1]
    
    # To print 100 images, split the string by both '\' and '.'
    if '\\' in img_path:
        target_string = re.split(r'[\\.]', img_path)[1]
    else:
        # to print the result one by one, split the string by both '/' and '.'
        target_string = img_path.split('.')[0]
        target_string = target_string.split('/')[1]
        
    # get annotation of img from txt file
    with open('TsignRecgTrain4170Annotation.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the target string is in the line
            if target_string in line:
                # If the string is found, get the line and spilt it
                line = line.strip()
                items = line.split(';')
    
    x1, y1, x2, y2 = int(items[3]), int(items[4]), int(items[5]), int(items[6])
    
    boxes = [[x1, y1, x2, y2], [x, y, w, h]]
    
    # resize the given bouding box so that the size is the same as the drawn bouding box
    gt = [int(i * factor) for i in boxes[0]]
    
    #convert to xyxy format
    pred = convert_xywh_to_xyxy(boxes[1])
    
    #draw ground truth
    cv.rectangle(img_copy, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 255), 2)
    cv.rectangle(img_gt, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 255), 2)
    
    #draw bounding box
    cv.rectangle(img_copy, (pred[0], pred[1]), (pred[2], pred[3]), (0, 255, 0), 2)
    
    #put text for iou img
    position = (15, 20)
    (text_width, text_height), _ = cv.getTextSize(f"IOU: {computeIOU(gt, pred):.3f}", cv.FONT_HERSHEY_COMPLEX, 0.7, 2)
    background_rect_coords = ((position[0], position[1] - text_height), 
                          (position[0] + text_width, position[1]))
    cv.rectangle(img_copy, background_rect_coords[0], background_rect_coords[1], (0, 0, 0), -1)
    cv.putText(img_copy, f"IOU: {computeIOU(gt, pred):.3f}", (15, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    #put text for gt img
    position = (15, 20)
    (text_width, text_height), _ = cv.getTextSize("Ground Truth", cv.FONT_HERSHEY_COMPLEX, 0.7, 2)
    background_rect_coords = ((position[0], position[1] - text_height), 
                          (position[0] + text_width, position[1]))
    cv.rectangle(img_gt, background_rect_coords[0], background_rect_coords[1], (0, 0, 0), -1)
    cv.putText(img_gt, "Ground Truth", (15, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    return computeIOU(gt, pred), img_copy, img_gt

def process_image(image_path):
    
    img = cv.imread(image_path)
    #preprosses img
    img_resize, img_blur, img_merged, img_gray = preprocessing(img)
    
    # get largest contour from color mask
    color_contour = color_seg(img, img_blur, img_gray)

    #segment using final contour (mask)
    extractReg = extractRegion(color_contour, img_resize)

    #draw bouding box
    bounded, x, y, w, h = bounding(color_contour, img_resize)

    #concat all img to be displayed
    concat_img = cv.hconcat([extractReg, bounded])
    cv.imshow(image_path, concat_img)
    
    k=cv.waitKey(0)
           
    cv.destroyAllWindows()
# Create a function to handle the file selection
def select_file():
    # Open a file dialog to choose an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    
    # Call your Python function with the selected file
    process_image(file_path)

# Create a tkinter window
window = tk.Tk()

# Set window title and size
window.title("Image Processing GUI")
window.geometry("400x200")

# Set background color
window.configure(bg="white")

# Create a label for instructions
label = tk.Label(window, text="Select an image file:", font=("Arial", 14), bg="white")
label.pack(pady=20)

# Create a button to trigger file selection
select_button = tk.Button(window, text="Select Image", command=select_file, font=("Arial", 12), bg="#007bff", fg="white")
select_button.pack(pady=10)

# Start the tkinter event loop
window.mainloop()