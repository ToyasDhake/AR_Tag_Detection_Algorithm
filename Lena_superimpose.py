from copy import deepcopy
from time import time, sleep
import Kalman
import cv2
import numpy as np

# Name of Video files and images
video_name_1 = 'Tag0.mp4'
video_name_2 = 'Tag1.mp4'
video_name_3 = 'Tag2.mp4'
video_name_4 = 'multipleTags.mp4'
image_name = 'Lena.png'
# Set VideoCapture
cap = cv2.VideoCapture(video_name_1)
# Read Image
image = cv2.imread(image_name)

list_KFs = []
for i in range(8):
    KF = Kalman.KF2d(dt=1)
    init_P = 1 * np.eye(4, dtype=np.float)
    init_x = np.array([0, 0, 0, 0], dtype=np.float)
    dict_KF = {'KF': KF, 'P': init_P, 'x': init_x}
    list_KFs.append(dict_KF)


# Tags all the corners of the marker as Top Left, Top Right, Bottom Left and Bottom Left.
def tag_corner(threshold, coordinate):
    count = 0
    top_left, top_right, bottom_left, bottom_right = 0, 0, 0, 0
    if coordinate[0] < 950 and coordinate[1] < 500:
        if threshold[coordinate[1] - 5][coordinate[0] - 5] == 255:
            count = count + 1
            top_left = 1
        if threshold[coordinate[1] + 5][coordinate[0] - 5] == 255:
            count = count + 1
            bottom_left = 1
        if threshold[coordinate[1] - 5][coordinate[0] + 5] == 255:
            count = count + 1
            top_right = 1
        if threshold[coordinate[1] + 5][coordinate[0] + 5] == 255:
            count = count + 1
            bottom_right = 1

        if count == 3:
            if top_left == 1 and top_right == 1 and bottom_left == 1:
                return True, 'TL'
            elif top_left == 1 and top_right == 1 and bottom_right == 1:
                return True, 'TR'
            elif top_left == 1 and bottom_right == 1 and bottom_left == 1:
                return True, 'BL'
            elif top_right == 1 and bottom_right == 1 and bottom_left == 1:
                return True, 'BR'
        else:
            return False, None
    else:
        return False, None

# Perform homography from one frame to another
def homography(corner1, corner2):
    index = 0
    M = np.empty((8, 9))
    for i in range(0, len(corner1)):
        x1 = corner1[i][0]
        y1 = corner1[i][1]
        x2 = corner2[i][0]
        y2 = corner2[i][1]
        M[index] = np.array([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        M[index + 1] = np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        index = index + 2

    U, s, V = np.linalg.svd(M, full_matrices=True)
    V = (deepcopy(V)) / (deepcopy(V[8][8]))
    H_matrix = V[8, :].reshape(3, 3)
    return H_matrix


# Convert image to AR Tag
def get_tag_matrix(img):
    height = img.shape[0]
    width = img.shape[1]
    bit_height = int(height / 8)
    bit_width = int(width / 8)
    m = 0
    ar_tag = np.empty((8, 8))
    for i in range(0, height, bit_height):
        n = 0
        for j in range(0, width, bit_width):
            black_count = 0
            white_count = 0
            for x in range(0, bit_height - 1):
                for y in range(0, bit_width - 1):
                    if img[i + x][j + y] == 0:
                        black_count = black_count + 1
                    else:
                        white_count = white_count + 1
            if white_count >= black_count:
                ar_tag[m][n] = 1
            else:
                ar_tag[m][n] = 0
            n = n + 1
        m = m + 1
    return ar_tag


# Get orientation of AR Tag
def get_tag_angle(tag):
    if tag[2][2] == 0 and tag[2][5] == 0 and tag[5][2] == 0 and tag[5][5] == 1:
        result = 0
    elif tag[2][2] == 1 and tag[2][5] == 0 and tag[5][2] == 0 and tag[5][5] == 0:
        result = 180
    elif tag[2][2] == 0 and tag[2][5] == 1 and tag[5][2] == 0 and tag[5][5] == 0:
        result = 90
    elif tag[2][2] == 0 and tag[2][5] == 0 and tag[5][2] == 1 and tag[5][5] == 0:
        result = -90
    else:
        result = None

    if result is None:
        return result, False
    else:
        return result, True


# Compute tag ID
def get_tag_id(tag_matrix):
    angle, flag = get_tag_angle(tag_matrix)
    if not flag:
        return flag, angle, None

    if flag:
        if angle == 0:
            id = tag_matrix[3][3] + tag_matrix[4][3] * 8 + tag_matrix[4][4] * 4 + tag_matrix[3][4] * 2
        elif angle == 90:
            id = tag_matrix[3][3] * 2 + tag_matrix[3][4] * 4 + tag_matrix[4][4] * 8 + tag_matrix[4][3]
        elif angle == 180:
            id = tag_matrix[3][3] * 4 + tag_matrix[4][3] * 2 + tag_matrix[4][4] + tag_matrix[3][4] * 8
        elif angle == -90:
            id = tag_matrix[3][3] * 1 + tag_matrix[3][4] + tag_matrix[4][4] * 2 + tag_matrix[4][3] * 4
        return flag, angle, id


# Convert matrix to list of x, y and z coordinates
def point_to_xyv(kp):
    kp = np.array(kp)
    x = kp[0::3].astype(int)
    y = kp[1::3].astype(int)
    v = kp[2::3].astype(int)
    return x, y, v

# Calculate time per frame
fps = cap.get(cv2.CAP_PROP_FPS)
time_per_frame = 1 / fps
previous_time = time()
# Buffer if algorithm misses calculation the AR Tag angle
last_angle = 0
impose_image = []

# Iterate through video
while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    # Check if frame exists
    if ret:
        # Pre-process to increase performace
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # Get contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        corners = []
        # Iterate through contours and look for possibility AR Tag
        for contour in contours:
            contour_corners = []
            if cv2.contourArea(contour) > 750:
                epsilon = 0.1 * cv2.arcLength(contour, True)
                approximation = cv2.approxPolyDP(contour, epsilon, True)

                for coordination in approximation:
                    result, key = tag_corner(threshold, [coordination[0][0], coordination[0][1]])
                    if result:
                        contour_corners.append([coordination[0][0], coordination[0][1], key])

                if len(contour_corners) == 4:
                    corners.append(contour_corners)
        # If Tag exists
        if corners:
            # Iterate through the corner tagging them
            for i in range(0, len(corners)):
                ar_tag_corners = [0, 0, 0, 0]
                for value in corners[i]:
                    if value[-1] == 'TL':
                        ar_tag_corners[0] = value[0:2]
                    elif value[-1] == 'TR':
                        ar_tag_corners[1] = value[0:2]
                    elif value[-1] == 'BL':
                        ar_tag_corners[2] = value[0:2]
                    elif value[-1] == 'BR':
                        ar_tag_corners[3] = value[0:2]
                # Draw circles at each corner of AR Tag
                if not (0 in ar_tag_corners):
                    # Calculate homography to transforms the tag from world to camera frame.
                    H = homography(ar_tag_corners, [[0, 0], [0, 47], [47, 0], [47, 47]])
                    H1 = np.linalg.inv(H)
                    # Create an empty image
                    ar_tag_image = np.zeros((48, 48))
                    # Copy the AR Tag tag in video to new  frame
                    for m in range(0, 48):
                        for n in range(0, 48):
                            x1, y1, z1 = np.matmul(H1, [m, n, 1])
                            if 540 > int(y1 / z1) > 0 and 960 > int(x1 / z1) > 0:
                                ar_tag_image[m][n] = threshold[int(y1 / z1)][int(x1 / z1)]
                    # Pre-process the newly created tag for better results
                    ar_tag_image = ar_tag_image.astype('float32')
                    ar_tag_image = cv2.medianBlur(ar_tag_image, 3)
                    kernel = np.ones((5, 5), np.uint8)
                    ar_tag_image = cv2.morphologyEx(ar_tag_image, cv2.MORPH_OPEN, kernel)
                    ar_tag_image = get_tag_matrix(ar_tag_image)
                    # Compute tag id
                    flag, angle_value, identity = get_tag_id(ar_tag_image)
                    # Set Lena to angle of the AR Tag
                    if flag:
                        last_angle = angle_value
                        if angle_value == 0:
                            lena_corners = ar_tag_corners
                        elif angle_value == 90:
                            lena_corners = [ar_tag_corners[2], ar_tag_corners[0], ar_tag_corners[3], ar_tag_corners[1]]
                        elif angle_value == -90:
                            lena_corners = [ar_tag_corners[1], ar_tag_corners[3], ar_tag_corners[0], ar_tag_corners[2]]
                        elif angle_value == 180:
                            lena_corners = [ar_tag_corners[3], ar_tag_corners[2], ar_tag_corners[1], ar_tag_corners[0]]

                        # Resize Lena
                        image = cv2.resize(image, (128, 128))
                        # Calculate homography matrx for Lena from world to camera frame
                        H1 = homography(lena_corners, [[0, 0], [0, 127], [127, 0], [127, 127]])
                        # Take inverse of that matrix
                        H1_inv = np.linalg.inv(H1)
                        impose_image = []
                        # Compute coordinate in final frame for each point in Lena
                        for j in range(0, 128):
                            for k in range(0, 128):
                                x2, y2, z2 = np.dot(H1_inv, [j, k, 1])
                                if 540 > int(y2 / z2) > 0 and 960 > int(x2 / z2) > 0:
                                    impose_image.append([int(y2 / z2), int(x2 / z2), image[j][k]])
                    # If algorithm misses the AR Tag
                    else:
                        angle_value = last_angle
                        # Set Lena to angle of the AR Tag
                        if angle_value == 0:
                            lena_corners = ar_tag_corners
                        elif angle_value == 90:
                            lena_corners = [ar_tag_corners[2], ar_tag_corners[0], ar_tag_corners[3], ar_tag_corners[1]]
                        elif angle_value == -90:
                            lena_corners = [ar_tag_corners[1], ar_tag_corners[3], ar_tag_corners[0], ar_tag_corners[2]]
                        elif angle_value == 180:
                            lena_corners = [ar_tag_corners[3], ar_tag_corners[2], ar_tag_corners[1], ar_tag_corners[0]]

                        # Resize Lena
                        image = cv2.resize(image, (128, 128))
                        # Calculate homography matrx for Lena from world to camera frame
                        H1 = homography(lena_corners, [[0, 0], [0, 127], [127, 0], [127, 127]])
                        # Take inverse of that matrix
                        H1_inv = np.linalg.inv(H1)
                        list_estimate = []
                        for i in range(8):
                            z = np.array([H1_inv[0][0], H1_inv[1][1]], dtype=np.float)
                            KF = list_KFs[i]['KF']
                            x = list_KFs[i]['x']
                            P = list_KFs[i]['P']

                            x, P, filtered_point = KF.process(x, P, z)

                            list_KFs[i]['KF'] = KF
                            list_KFs[i]['x'] = x
                            list_KFs[i]['P'] = P

                            v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 2

                            list_estimate.extend(list(filtered_point) + [v])
                        fx, fy, fv = point_to_xyv(list_estimate)
                        impose_image = []
                        # Compute coordinate in final frame for each point in Lena
                        for j in range(0, 128):
                            for k in range(0, 128):
                                x2, y2, z2 = np.dot(H1_inv, [j, k, 1])
                                if 540 > int(y2 / z2) > 0 and 960 > int(x2 / z2) > 0:
                                    impose_image.append([int(y2 / z2), int(x2 / z2), image[j][k]])
                    # Impose Lena on final output frame
                    for pixel in impose_image:
                        frame[pixel[0]][pixel[1]] = pixel[2]
        # Display output frame
        cv2.imshow('Frame', frame)
        # Quit if user press escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Limit the video playback to the original video frame rate
        new_time = time()
        if new_time - previous_time < time_per_frame:
            sleep(time_per_frame - (new_time - previous_time))
        previous_time = new_time
    # Break if frame is not available
    else:
        break
# Release VideoCapture and destroy open windows
cap.release()
cv2.destroyAllWindows()
