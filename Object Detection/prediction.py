# load necessary libraries
from keras.models import load_model
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from yolo3_one_file_to_detect_them_all import decode_netout
from yolo3_one_file_to_detect_them_all import correct_yolo_boxes
from yolo3_one_file_to_detect_them_all import do_nms
from yolo3_one_file_to_detect_them_all import draw_boxes


# load and prepare the image
def load_image(path, shape):
    img = load_img(path)
    width, height = img.size

    img = load_img(path, target_size = shape)
    img = img_to_array(img)
    img = img.astype('float32')
    img /= 255.0
    img = expand_dims(img, 0)

    return img, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

# load the model
model = load_model("G:/rauf/STEPBYSTEP/Data2/PRE-TRAINED/weights/model.h5")

# path
img_path = "G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/CV/Object Detection/zebra.jpg"

# load image
width, height = 416, 416
img, img_w, img_h = load_image(img_path, (width, height))

# predict
y_that = model.predict(img)

# print the box values as inside numpy array
#print(y_that)

# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
class_threshold = 0.6
boxes = list()
for i in range(len(y_that)):
    boxes += decode_netout(y_that[i][0], anchors[i], class_threshold, width, height)

correct_yolo_boxes(boxes, img_h, img_w, height, width)
do_nms(boxes, 0.5)

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
draw_boxes(img_path, v_boxes, v_labels, v_scores)

# Waw awesome, manage it.
# finally we created the model which can identify object from image
# rauf odilov keep training