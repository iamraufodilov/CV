# load libraries
import yolo3_one_file_to_detect_them_all

# load the model
model = yolo3_one_file_to_detect_them_all.make_yolov3_model()

# load the weights into the model
weight_reader = yolo3_one_file_to_detect_them_all.WeightReader("G:/rauf/STEPBYSTEP/Data2/PRE-TRAINED/weights/yolov3.weights")

weight_reader.load_weights(model)
model.save("G:/rauf/STEPBYSTEP/Data2/PRE-TRAINED/weights/model.h5")