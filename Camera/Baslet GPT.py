import cv2
from pypylon import pylon

# Set up the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set up the camera properties
camera.PixelFormat = "Mono8"
camera.GainAuto = "Off"
camera.ExposureAuto = "Off"
camera.ExposureTime = 10000

# Define the region of interest (ROI)
x, y, w, h = cv2.selectROI("Select ROI", camera.GrabOne().Array)
cv2.destroyAllWindows()

# Start the video capture
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Process the frames
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Convert the frame to grayscale
        frame = grabResult.Array
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop the frame to the ROI
        cropped = gray[y:y+h, x:x+w]

        # Threshold the frame to separate the foreground from the background
        _, thresh = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

        # Find the contours in the thresholded frame
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assuming it is the mouse)
        if len(contours) > 0:
            maxContour = max(contours, key=cv2.contourArea)
            M = cv2.moments(maxContour)

            if M["m00"] > 0:
                # Calculate the center of the contour
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw a circle at the center of the contour
                cv2.circle(frame, (cx+x, cy+y), 5, (0, 255, 0), -1)

                # Display the frame
                cv2.imshow("Frame", frame)

        # Wait for a key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabResult.Release()

# Stop the video capture and close all windows
camera.StopGrabbing()
cv2.destroyAllWindows()