import cv2 
import numpy as np 
import tempfile
import os
import sys

  
numberOfFiducials = 3
fiducialOuterDiameterInPixels = 18
fiducialInnerDiameterInPixels = 11


# Function to compute distance between two points
def distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def sort_points(norm_separation_fid, norm_separation_kp):
    cv2.waitKey()
   # for n in norm_separation_fid:
        #for m in norm_separation_kp:
           # compare = norm_separation_fid[n] - norm_separation_kp[m]
            #if compare <= difference_thresh:
                #difference m is actually between points n

# Create a synthetic fiducial image
pattern_sizeHW = [fiducialOuterDiameterInPixels+5, fiducialOuterDiameterInPixels+5]

if fiducialOuterDiameterInPixels %2 == 0:  # Make sure the pattern size is odd
    pattern_sizeHW[0] += 1
    pattern_sizeHW[1] += 1
fiducial_pattern = np.ones(pattern_sizeHW, dtype=np.uint8)*165
cv2.circle(fiducial_pattern, (pattern_sizeHW[1]//2, pattern_sizeHW[0]//2), fiducialOuterDiameterInPixels//2, 70, cv2.FILLED)  # The outer disk is dark gray
cv2.circle(fiducial_pattern, (pattern_sizeHW[1]//2, pattern_sizeHW[0]//2), fiducialInnerDiameterInPixels//2, 255, cv2.FILLED)  # The inner disk is white

# Standardize the pattern image
standardized_fiducial_pattern = (fiducial_pattern.astype(np.float32) - fiducial_pattern.mean())/fiducial_pattern.std()
standardized_fiducial_pattern_float32 = standardized_fiducial_pattern.astype(np.float32)
#cv2.imshow('test',standardized_fiducial_pattern_float32)
cv2.waitKey()
cv2.imshow('standardized_fiducial_pattern',standardized_fiducial_pattern)


cap = cv2.VideoCapture(1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Save the current frame to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_frame_file:
        frame_filename = temp_frame_file.name
        cv2.imwrite(frame_filename, frame)  # Save frame as a PNG image
        
    img_cam = cv2.imread(frame_filename, cv2.IMREAD_GRAYSCALE)
    hT,wT = img_cam.shape
    #DXF = cv2.resize(cv2.imread('output_SingleLayer.png', cv2.IMREAD_GRAYSCALE),(hT,wT))
    
    #cv2.imshow('test',img_cam)
    #cv2.waitKey()
    
    if img_cam is None:
        print("Error: Could not open or find the PCB image.")
    else:
        print("PCB image successfully loaded.")
    img_shapeHWC = [0,0]
    img_shapeHWC[0],img_shapeHWC[1],= img_cam.shape
    img_cam_float32 = img_cam.astype(np.float32)
    print(f"Image Type: {img_cam_float32.dtype}")
    print(f"Template Type: {standardized_fiducial_pattern.dtype}")
    # Pattern match
    match_img = cv2.matchTemplate(img_cam_float32, standardized_fiducial_pattern_float32, cv2.TM_CCOEFF_NORMED)
    # Create an 8-bit version of the match image for visualization, padded with zeros to get an image the same size as the original
    padded_match_8bits_img = np.zeros((img_shapeHWC[0], img_shapeHWC[1]), dtype=np.uint8)
    padded_match_8bits_img[fiducial_pattern.shape[0]//2: fiducial_pattern.shape[0]//2 + match_img.shape[0],
        fiducial_pattern.shape[1]//2: fiducial_pattern.shape[1]//2 + match_img.shape[1]] = (128 * (match_img + 1.0)).astype(np.uint8)
    
    #cv2.imshow('test',padded_match_8bits_img)
    #cv2.waitKey()
    
    # Find the optimal threshold to detect the expected number of fiducials
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10  # Minimum area for a blob
    params.maxArea = 1000  # Maximum area for a blob
    params.filterByCircularity = True
    params.minCircularity = 0.90  # Minimum circularity of blobs
    params.filterByInertia = False
    params.minInertiaRatio = 0.1  # Minimum inertia ratio of blobs
    optimal_threshold = None
    optimal_seedPoint_boundingBox_list = None
    optimal_annotated_blobs_img = None
    detector = cv2.SimpleBlobDetector_create(params)
    _, thresholded_img = cv2.threshold(padded_match_8bits_img, 190, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(cv2.bitwise_not(thresholded_img))
    print(keypoints)
    blobs_img = cv2.drawKeypoints(cv2.bitwise_not(thresholded_img), keypoints, np.array([]), (255, 0, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    Final_detections = cv2.drawKeypoints(img_cam, keypoints, np.array([]), (255, 0, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    Final_detections_float32 = Final_detections.astype(np.float32)
    #cv2.imshow('blobs',blobs_img)
    #cv2.waitKey()
    cv2.imshow('Final_detections',Final_detections)
    
    #orb = cv2.ORB_create(nfeatures=1000)
    #kp1, des1 = orb.detectAndCompute(thresholded_img,None)
    fiducials = ((271.654,3779.528),(3669.291,3740.157),(204.724,996.063))
    
    separation_kp = {}
    max_separation_kp = 0
    separation_fid = {}
    max_separation_fid = 0
    difference_thresh = 0.01
    
    # Loop through each pair of keypoints to calculate distances
    for i, pt1 in enumerate(keypoints):
        for j, pt2 in enumerate(keypoints):
            if i > j:  # Avoid comparing the keypoint with itself
                pt1_coords = pt1.pt  # Coordinates of the first keypoint
                pt2_coords = pt2.pt  # Coordinates of the second keypoint
                # Calculate the distance between pt1 and pt2
                dist = distance(pt1_coords, pt2_coords)
                separation_kp[(i, j)] = dist
                if dist > max_separation_kp:
                    max_separation_kp = dist
                    
    #print(separation_kp)
    #normalize separation array
    norm_separation_kp = {}
    for element in separation_kp:
        norm_separation_kp[element] = separation_kp[element]/max_separation_kp
     
         
    for n, pt1 in enumerate(fiducials):
        for m, pt2 in enumerate(fiducials):
            if(n > m):
                pt1_coords = pt1  # Coordinates of the first keypoint
                pt2_coords = pt2  # Coordinates of the second keypoint
                # Calculate the distance between pt1 and pt2
                dist = distance(pt1_coords, pt2_coords)
                separation_fid[(n, m)] = dist
                if dist > max_separation_fid:
                    max_separation_fid = dist
                    
    #print(separation_fid)
    #normalize separation array
    norm_separation_fid = {}
    for element in separation_fid:
        norm_separation_fid[element] = separation_fid[element]/max_separation_fid
    
    
    #sort_points(norm_separation_fid,norm_separation_kp)
    
    print(norm_separation_fid) 
    print(norm_separation_kp)
            
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    Final_detections = cv2.drawKeypoints(img_cam, kp1, np.array([]), (255, 0, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgFeatures = cv2.drawMatches(DXF,kp1,img_cam,kp2,good,None,flags=2)
    # Resize the resulting image to fit the screen
    height, width = imgFeatures.shape[:2]
    max_size = 800  # Set the max size for the display
    scaling_factor = max_size / float(width)  # Calculate the scaling factor
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_imgMatches = cv2.resize(imgFeatures, (new_width, new_height))
    
    cv2.namedWindow("Features", cv2.WINDOW_NORMAL)
    cv2.imshow('Features',resized_imgMatches)
    cv2.waitKey()
    print(kp1[0].pt)
    print(des1)
       ''' 
    # Check for keypress without blocking the frame update, and exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting loop...")
        break
    
    os.remove(frame_filename)
# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()