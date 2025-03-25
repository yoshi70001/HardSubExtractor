import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf, verbose=False)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    # print(results)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


model = YOLO("yolo12n.pt",verbose=False)
model = model.cuda()
input_filename = "videos/AormeSubs_Araiya_san!_Ore_to_Aitsu_ga_Onnayu_de!_01_SIN_CENSURA.mp4"
cap = cv2.VideoCapture(input_filename,cv2.CAP_FFMPEG)
heigth = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# writer = create_video_writer(cap, output_filename)
counterFrame = 1
while True:
    success, img = cap.read()
    if not success:
        break
    if counterFrame%5==0:
        img = cv2.resize(img,(int(width*(360/heigth)),360),interpolation=cv2.INTER_AREA)
        result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
        # writer.write(result_img)
        cv2.imshow("Image", img)
        cv2.imshow("result_img", result_img)
        
        cv2.waitKey(1)
    counterFrame+=1
cap.release()
# writer.release()