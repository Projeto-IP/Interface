import streamlit as st
import cv2
from ultralytics import YOLO
import math

# Classes de objetos (neste caso, apenas Coca-Cola)
classNames = ["Coca-Cola"]

# Carregar o modelo YOLOv8 treinado
def load_model(model_path):
    try:
        model = YOLO(model_path)
        st.success("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para realizar a detecção de objetos em um frame
def detect_objects(frame, model):
    results = model(frame, stream=True)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]
            detections.append((label, confidence, (x1, y1, x2, y2)))
    return detections

# Função para exibir os resultados de detecção no frame
def draw_detections(frame, detections):
    for label, confidence, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, f'{label} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

# Função para capturar e processar o vídeo em tempo real
def video_detection(model):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Falha ao capturar a imagem da câmera.")
            break

        detections = detect_objects(frame, model)
        
        if detections:
            st.write(f"Detecções: {detections}")
        else:
            st.write("Nenhuma detecção.")

        frame = draw_detections(frame, detections)
        stframe.image(frame, channels="BGR")

    cap.release()

# Interface do Streamlit
st.title("Detecção de Latas de Coca-Cola em Tempo Real com YOLOv8")
st.text("Usando YOLOv8 para detectar latas de Coca-Cola em tempo real")

model_path = st.text_input("Caminho para o modelo YOLOv8 (.pt)", value="C:/Igor/best.pt")

if st.button("Iniciar Detecção"):
    model = load_model(model_path)
    if model:
        video_detection(model)
