import cv2
from pypylon import pylon
import numpy as np

# Connect to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Set camera parameters
camera.Open()
camera.PixelFormat = "Mono8"

# Create window for selecting ROI
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)

# Select region of interest
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
if grab_result.GrabSucceeded():
    img = grab_result.Array
    roi = cv2.selectROI("Select ROI", img)
    cv2.destroyWindow("Select ROI")

    # Set region of interest as image mask
    x, y, w, h = roi
    mask = np.zeros((camera.Height.Value, camera.Width.Value), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
else:
    raise RuntimeError("Failed to grab initial frame.")

# Start grabbing images

while camera.IsGrabbing():
    # Wait for the next image
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    # Check if image was successfully retrieved
    if grab_result.GrabSucceeded():
        # Get image as a numpy array
        img = grab_result.Array

        # Apply mask to image
        img = cv2.bitwise_and(img, img, mask=mask)

        # Threshold image to separate foreground and background
        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # Find contours in image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box and calculate center of each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)

        # Display image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        # Exit on ESC key press
        if key == 27:
            break

    # Release grab result
    grab_result.Release()

# Stop grabbing images and close window
camera.StopGrabbing()
cv2.destroyAllWindows()
