import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# Load and Process Calibration Image
calib_path = 'calibration_grid_26065.png'
calib_img = cv2.imread(calib_path)
if calib_img is None:
    raise IOError(f"Cannot open calibration image: {calib_path}")

gray_calib = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
gray_calib = cv2.equalizeHist(gray_calib)
gray_calib_cropped = gray_calib[:, 200:]
color_calib_cropped = calib_img[:, 200:]

pattern_size = (20, 16)
square_size_mm = 10

ret, corners = cv2.findChessboardCorners(
    gray_calib_cropped,
    pattern_size,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

if not ret:
    raise RuntimeError("Checkerboard corners not detected. Check pattern size and image quality.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray_calib_cropped, corners, (11, 11), (-1, -1), criteria)

pixel_dists = []
for i in range(pattern_size[0] - 1):
    pt1 = corners[i][0]
    pt2 = corners[i + 1][0]
    dist = np.linalg.norm(pt2 - pt1)
    pixel_dists.append(dist)

avg_pixel_per_10mm = np.mean(pixel_dists)
mm_per_pixel = square_size_mm / avg_pixel_per_10mm
print(f"[INFO] Calibration complete: {mm_per_pixel:.4f} mm/pixel")

# Show calibration result
vis = color_calib_cropped.copy()
cv2.drawChessboardCorners(vis, pattern_size, corners, ret)

# Load Bubble Image
image_path = 'frame_0306.png'
image = cv2.imread(image_path)
if image is None:
    raise IOError(f"Cannot open image file: {image_path}")

# Select ROI Interactively
roi_rect = cv2.selectROI("Select ROI", image, showCrosshair=True)
cv2.destroyAllWindows()

x_start, y_start, width, height = roi_rect
x_end, y_end = x_start + width, y_start + height
roi = image[y_start:y_end, x_start:x_end]

# Image Processing Pipeline
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Sharpen
blurred = cv2.GaussianBlur(gray, (9, 9), 2)
sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

# Blur before threshold
blur = cv2.GaussianBlur(sharpened, (5, 5), 0)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Morphological opening (reduced to preserve shape)
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Use connected components from cleaned binary mask
binary_mask = opened
markers = label(binary_mask)

# Bubble Detection and Annotation
annotated_image = image.copy()
bubble_sizes_mm = []
min_area = 50
max_area = 10000

for region in regionprops(markers):
    area = region.area
    if min_area < area < max_area:
        eq_diam_pix = np.sqrt(4 * area / np.pi)
        eq_diam_mm = eq_diam_pix * mm_per_pixel
        bubble_sizes_mm.append(eq_diam_mm)

        y, x = region.centroid
        center = (int(x) + x_start, int(y) + y_start)
        radius = int(eq_diam_pix / 2)
        cv2.circle(annotated_image, center, radius, (0, 255, 0), 2)

# Visualization
plt.figure(figsize=(6, 5))
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened ROI")
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(thresh, cmap='gray')
plt.title("Adaptive Threshold")
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(opened, cmap='gray')
plt.title("Binary Mask")
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Bubbles")
plt.axis('off')
plt.show()

# Histogram
bubble_sizes_mm = np.array(bubble_sizes_mm)
plt.figure(figsize=(8, 5))
plt.hist(bubble_sizes_mm, bins=40, density=True, edgecolor='black', alpha=0.7)
plt.xlabel('Equivalent Bubble Diameter (mm)')
plt.ylabel('Probability Density')
plt.title('PDF of Bubble Sizes (in mm)')
plt.grid(True)
plt.show()
