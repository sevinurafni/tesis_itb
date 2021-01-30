import cv2
import numpy as np
from scipy import stats


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gradient_direction)
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    return binary_output


def sobel_filter(image, ksize=3):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    s_channel = hls[:, :, 0]

    gradx = abs_sobel_thresh(s_channel, 'x', 10, 200)
    grady = abs_sobel_thresh(s_channel, 'y', 10, 200)

    combined = np.zeros_like(grady)
    combined_condition = ((gradx == 1) & (grady == 1))
    return combined_condition

def hls_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Threshold color channel
    s_thresh_min = 120
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition




def hsv_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_thresh_min = 160
    s_thresh_max = 255
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition




def yuv_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    s_channel = hls[:, :, 0]

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition


def rgb_filter(image):
    # Extract RG colors for better yellow line isolation
    color_threshold = 170
    R = image[:, :, 0]
    G = image[:, :, 1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)
    return r_g_condition


def filter_image(image, is_blind=False):
    sobel_condition = sobel_filter(image)
    hls_condition = hls_filter(image)
    rgb_condition = rgb_filter(image)
    hsv_condition = hsv_filter(image)
    yuv_condition = yuv_filter(image)

    height, width = image.shape[0], image.shape[1]
    # apply the region of interest mask
    combined_binary = np.zeros((height, width), dtype=np.uint8)
    if not is_blind:
        combined_binary[((rgb_condition | hsv_condition | yuv_condition) & (hls_condition | sobel_condition))] = 1
    else :
        combined_binary[sobel_condition] = 1

    mask = np.zeros_like(combined_binary)
    region_of_intersect = np.array([[0, height], [width / 2, int(0.5 * height)], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_intersect], 1)
    thresholded = cv2.bitwise_and(combined_binary, mask)
    return thresholded

def get_offset_from_center(left_x, right_x, height=720, width=1280):

    lane_center = (right_x[height-1] + left_x[height-1]) / 2
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    img_center_offset = abs(width / 2 - lane_center)
    offset_metters = xm_per_pix * img_center_offset
    return offset_metters


def get_source_points():
    return [[220,720], [1100, 720], [780, 470], [600, 470]]


def get_destination_points(width, height, fac=0.3):
    fac = 0.3
    p1 = [fac * width, height]
    p2 = [width - fac * width, height]
    p3 = [width - fac * width, 0]
    p4 = [fac * width, 0]
    destination_points = [p1,p2,p3,p4]
    return destination_points


def perspective_transform(image):
    height, width = image.shape[0], image.shape[1]
    img_size = (width,height)
    source_points = get_source_points()
    destination_points = get_destination_points(width, height)
    src = np.float32(source_points)
    dst = np.float32(destination_points)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def perspective_transform_with_filled_area(original_image, filtered_image):
    warped = perspective_transform(filtered_image)
    source_points = np.array(get_source_points())
    filled = cv2.polylines(original_image.copy(), [source_points], True, (0, 255, 0), thickness=2)
    return warped, filled

