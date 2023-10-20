import cv2
import time

# https://github.com/ronitsinha/speed-detector/tree/master

# Constants
WAIT_TIME = 1

# Colors for drawing on processed frames
DIVIDER_COLOR = (255, 255, 0)
BOUNDING_BOX_COLOR = (255, 0, 0)
CENTROID_COLOR = (0, 0, 255)

def filter_mask (image):
	kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
	kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

	# Remove noise
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
	# Close holes within contours
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
	# Merge adjacent blobs
	dilation = cv2.dilate(closing, kernel_dilate, iterations = 2)

	return dilation

def get_centroid (x, y, w, h):
	x1 = w // 2
	y1 = h // 2

	return(x+x1, y+y1)

def detect_vehicles (image):

	MIN_CONTOUR_WIDTH = 10
	MIN_CONTOUR_HEIGHT = 10

	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	matches = []

	# Hierarchy stuff:
	# https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy
	for (i, contour) in enumerate(contours):
		x, y, w, h = cv2.boundingRect(contour)
		contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

		if not contour_valid or not hierarchy[0,i,3] == -1:
			continue

		centroid = get_centroid(x, y, w, h)

		matches.append( ((x,y,w,h), centroid) )

	return matches

def process_frame(frame_number, frame, bg_subtractor, car_counter):
	processed = frame.copy()
	gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

	# if car_counter.is_horizontal:
	# 	cv2.line(processed, (0, car_counter.divider), (frame.shape[1], car_counter.divider), DIVIDER_COLOR, 1)
	# else:
	# 	cv2.line(processed, (car_counter.divider, 0), (car_counter.divider, frame.shape[0]), DIVIDER_COLOR, 1)

	fg_mask = bg_subtractor.apply(gray)
	fg_mask = filter_mask(fg_mask)
	matches = detect_vehicles(fg_mask)

	for (i, match) in enumerate(matches):
		contour, centroid = match

		x,y,w,h = contour

		cv2.rectangle(processed, (x,y), (x+w-1, y+h-1), BOUNDING_BOX_COLOR, 1)
		cv2.circle(processed, centroid, 2, CENTROID_COLOR, -1)

	car_counter.update_count(matches, frame_number, processed)

	return processed

def main():
	
    # TODO: Add video file path here...
    videoFile = ""

    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
	# car_counter = None

    cap = cv2.VideoCapture(videoFile)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    cv2.namedWindow('Source Image')

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_number = -1

    while True:
        frame_number += 1
        ret, frame = cap.read()

        if not ret:
            print('Frame capture failed, stopping...')
            break

        if car_counter is None:
			# TODO: Implement car_counter
            car_counter = ''
            #car_counter = VehicleCounter(frame.shape[:2], road, cap.get(cv2.CAP_PROP_FPS), samples=10)

        processed = process_frame(frame_number, frame, bg_subtractor, car_counter)

        cv2.imshow('Source Image', frame)
        cv2.imshow('Processed Image', processed)

		# Keep video's speed stable
        time.sleep( 1.0 / cap.get(cv2.CAP_PROP_FPS) )


    print('Closing video capture...')
    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
	main()