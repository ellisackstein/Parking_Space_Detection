import cv2
import numpy as np

# Read the image
image = cv2.imread('Corolla.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area, aspect ratio, etc.
min_area = 1000  # Example threshold for minimum contour area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Create a blank mask
mask = np.zeros_like(gray)

# Draw contours on the mask
cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

# Apply mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the masked image
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()