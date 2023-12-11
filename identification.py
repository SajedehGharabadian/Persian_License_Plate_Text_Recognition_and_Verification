import cv2
from ultralytics import YOLO
from deep_text_recognition_benchmark.dtrb import DTRB
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
# parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--detector_weights',type=str,default='weights/yolov8_detector/yolo-v8-n.pt')
parser.add_argument('--recognizer_weights',type=str,default='weights/dtrb_recognizer/dtrb_resnet_bilstm_ctc.pth')
parser.add_argument('--input_img',type=str)
parser.add_argument('--threshold',type=float,default=0.7)
#,default='io/input/photo_2023-11-07_14-39-15.jpg'
opt = parser.parse_args()


plate_detector = YOLO(opt.detector_weights)

plate_recognizer = DTRB(weights_path=opt.recognizer_weights,opt=opt)


img = cv2.imread(opt.input_img)
result = plate_detector(img)
result = result[0]

# dh, dw, _ = img.shape
# i = 0

# file = open('runs\detect\predict\labels\image0.txt','r')
# for line in file:
#     if result.boxes.conf > opt.threshold:
#         print(result.boxes.conf)
#         class_id, x_center, y_center, w, h = line.strip().split()
#         x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
#         x_center = round(x_center * dw)
#         y_center = round(y_center * dh)
#         w = round(w * dw)
#         h = round(h * dh)
#         x = round(x_center - w / 2)
#         y = round(y_center - h / 2)

#         imgCrop = img[y:y + h, x:x + w]
#         cv2.imwrite('io/output/image_crop'+str(i)+'.jpg',imgCrop)
#         plate_img = cv2.resize(imgCrop,(100,32))
#         plate_img = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
#         plate_recognizer.predict(plate_img,opt)
#         i += 1
#     else:
#         print('License plate not detect')

# cv2.imwrite('io/output/image.jpg',img)

for i in range(len(result.boxes.xyxy)):  # tedad pelak ha dar yek aks
    print(result.boxes.conf[i])
    if result.boxes.conf[i] > opt.threshold:
        bbox = result.boxes.xyxy[i]  #tensor
        print(bbox)
        bbox = bbox.cpu().detach().numpy().astype(int) #ndarray
        crop_plate_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
        cv2.imwrite("io/output/crop_img"+str(i)+".jpg", crop_plate_img)
        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),4)

        plate_img = cv2.resize(crop_plate_img,(100,32))
        plate_img = cv2.cvtColor(crop_plate_img,cv2.COLOR_BGR2GRAY)
        d = plate_recognizer.predict(plate_img,opt=opt)
        print(d)
        
        


