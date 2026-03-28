import cv2
import torch
import torch.nn.functional as F
import webbrowser
import time

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from model import load_tivit_model, transform_frame

#подгонка под gradcam
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=load_tivit_model(config.WEIGHTS_PATH, device)
    
    target_layers=[model.encoder.layers[-1].ln_1]
    cam=GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    cap = cv2.VideoCapture(0)
    #детектор лиц
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    last_opened_time=20
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Нет доступа к камере")
            break
            
        #чб
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Если лицо найдено
        if len(faces)>0:
            x, y, w, h = faces[0]
            face_roi=frame[y:y+h, x:x+w]
            face_resized=cv2.resize(face_roi, (224,224))
            
            # rgb и трансформация
            rgb_face=cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_tensor=transform_frame(rgb_face)
            input_batch=input_tensor.unsqueeze(0).to(device)
            
            # маска внимания
            grayscale_cam=cam(input_tensor=input_batch, targets=None)[0, :]
            face_float=rgb_face.astype(np.float32)/255.0
            #лицо + heatmap
            cam_image=show_cam_on_image(face_float, grayscale_cam, use_rgb=True)
            #2bgr
            cam_image_bgr=cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cam_image_resized=cv2.resize(cam_image_bgr, (w, h))
            frame[y:y+h, x:x+w]=cam_image_resized
            
            with torch.no_grad():
                output = model(input_batch)
                probabilities = F.softmax(output[0], dim=0)
                drowsy_prob = probabilities[0].item()
        else:
            drowsy_prob = 0.0 
        
        cv2.putText(frame, f"Tired: {drowsy_prob*100:.1f}%", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.time()
        if drowsy_prob > config.CONFIDENCE_THRESHOLD:
            if (current_time - last_opened_time)>config.COOLDOWN_SECONDS:
                webbrowser.open(config.TARGET_URL)
                last_opened_time=current_time
        cv2.imshow('TiViT', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
