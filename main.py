import cv2
from ultralytics import YOLO
from deep_text_recognition_benchmark.dtrb import DTRB


plate_detector = YOLO('weights/yolov8_detector/yolo-v8-n.pt')

plate_recognizer = DTRB('weights/dtrb_recognizer/dtrb_resnet_bilstm_ctc.pth')


img = cv2.imread('io/input/photo_2023-11-07_14-39-15.jpg')
result = plate_detector(img, save_txt=True)
result = result[0]

dh, dw, _ = img.shape
i = 0

file = open('runs\detect\predict\labels\image0.txt','r')
for line in file:
    class_id, x_center, y_center, w, h = line.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)
    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    imgCrop = img[y:y + h, x:x + w]
    cv2.imwrite('io/output/image_crop'+str(i)+'.jpg',imgCrop)
    plate_img = cv2.resize(imgCrop,(100,32))
    plate_img = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
    plate_recognizer.predict(plate_img)
    i += 1


# for i in range(len(result.boxes.xyxy)):
#     print(result.boxes.conf[i])
#     if result.boxes.conf[i] > 0.65:
#         bbox = result.boxes.xyxy[i]  #tensor
#         print(bbox)
#         bbox = bbox.cpu().detach().numpy().astype(int) #ndarray
#         crop_plate_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
#         cv2.imwrite("io/output/crop_img"+str(i)+".jpg", crop_plate_img)
#         cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),4)

#         plate_img = cv2.resize(crop_plate_img,(100,32))
#         plate_img = cv2.cvtColor(crop_plate_img,cv2.COLOR_BGR2GRAY)
#         d = plate_recognizer.predict(plate_img)
#         print(d)
        
        


