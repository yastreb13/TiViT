import cv2
import torch
import torch.nn.functional as F
import webbrowser
import time

import config
from model import load_tivit_model, transform_frame

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=load_tivit_model(config.WEIGHTS_PATH, device)

    cap=cv2.VideoCapture(0)
    
    last_opened_time=20
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Нет достуа к камере")
            break
            
        # rgb
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # сжимаем до необходимого размера в трансформер
        input_tensor=transform_frame(rgb_frame)
        input_batch=input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output=model(input_batch)
            probabilities=F.softmax(output[0], dim=0)
            drowsy_prob=probabilities[0].item()
            
        
        cv2.putText(frame, f"Tired: {drowsy_prob*100:.1f}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        current_time = time.time()
        if drowsy_prob > config.CONFIDENCE_THRESHOLD:
            if (current_time-last_opened_time)>config.COOLDOWN_SECONDS:
                webbrowser.open(config.TARGET_URL)
                last_opened_time=current_time

        cv2.imshow('TiViT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()