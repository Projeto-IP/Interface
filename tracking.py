# Bibliotecas
import streamlit as st  # Interface
import cv2  # Uso de câmeras e manipulação de imagens
from ultralytics import YOLO  # Carregar modelo
import numpy as np  # Transformar imagens em matrizes
import math  # Cálculos
import smtplib  # E-mail
import email.message  # Texto e-mail
from email.mime.image import MIMEImage  # Para anexar a imagem no e-mail
from email.mime.multipart import MIMEMultipart  # Para criar e-mails com múltiplas partes

# Variável global
meta = 5

# Envia um e-mail padronizado automaticamente
def enviar_email(image_path):
    corpo_email = """
    Olá Sr. Repositor,
    O lote de Coca-cola da nossa prateleira acabou, por favor repôr o quanto antes para evitar perda de vendas!!!
    Segue em anexo uma imagem da prateleira no momento atual.
    """

    msg = MIMEMultipart()
    msg['Subject'] = "Prateleira com poucos produtos"
    msg['From'] = 'igordiasaguiar05@gmail.com'
    msg['To'] = 'dias2005aguiar@gmail.com'
    password = 'zgbrysnrcsrvibaj'

    msg.attach(email.message.Message())
    msg.get_payload()[0].set_payload(corpo_email)

    with open(image_path, 'rb') as file:
        img = MIMEImage(file.read())
        img.add_header('Content-Disposition', 'attachment', filename="prateleira.jpg")
        msg.attach(img)

    s = smtplib.SMTP('smtp.gmail.com: 587')
    s.starttls()
    s.login(msg['From'], password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string().encode('utf-8'))
    print('Email enviado')

# Classes de objetos (Rótulos)
classNames = ["Coca-Cola"]

# Carregar o modelo YOLOv8 treinado a partir do caminho dado pelo usuário
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para realizar a detecção de objetos em um frame
def detect_objects(frame, model):
    try:
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
    except Exception as e:
        st.error(f"Erro ao detectar objetos: {e}")
        return []

# Função para exibir os resultados de detecção no frame
def draw_detections(frame, detections):
    try:
        for label, confidence, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, f'{label} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame
    except Exception as e:
        st.error(f"Erro ao desenhar detecções: {e}")
        return frame

# Função para contar os objetos detectados
def count_objects(detections):
    object_count = len(detections)
    st.write(f"Número de objetos detectados: {object_count}")
    return object_count

## ESTRUTURA DO STREAMLIT

# Interface do Streamlit
st.title("Eye-Market")
st.text("Identificador de produtos para monitorar sua prateleira")

# Colocando o modelo de detecção utilizado no app atrelado ao que for passado no input do app
model_path = st.text_input("Caminho para o modelo YOLOv8 treinado (.pt)", value="C:\\Igor\\best.pt")

# Espaço para dar upload de arquivos 
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

#Cria botão para iniciar a contagem de objetos 
if st.button("Detectar Objetos"):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        model = load_model(model_path)
        
        if model:
            detections = detect_objects(frame, model)
            frame = draw_detections(frame, detections)
            object_count = count_objects(detections)

            st.image(frame, channels="BGR")

            if object_count < meta:
                print("Prateleira com poucos produtos!!!")
                st.warning("Os produtos estão acabando!")
                # Salvar a imagem para enviar por e-mail
                image_path = "prateleira.jpg"
                cv2.imwrite(image_path, frame)
                enviar_email(image_path)
        else:
            st.error("Erro ao carregar o modelo.")
    else:
        st.error("Por favor, carregue uma imagem.")
