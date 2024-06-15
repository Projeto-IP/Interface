import streamlit as st
import cv2
import numpy as np

# Função para carregar o vídeo e mostrar a detecção em tempo real
def video_detection():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Mude para o índice do seu dispositivo de captura de vídeo

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Falha ao capturar o vídeo")
            break
        
        # Aqui você deve adicionar o código da sua detecção usando YOLOv8
        # Processamento de detecção de objetos

        # Exibe o frame no Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()

# Função para mostrar o estoque em tempo real
def show_stock():
    # Aqui você deve adicionar o código para mostrar o estoque em tempo real
    stock_data = {
        'item1': 20,
        'item2': 35,
        'item3': 10
    }
    st.table(stock_data)

# Configurações da página do Streamlit
st.title("Sistema de Detecção de Imagens e Monitoramento de Estoque")

# Layout
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Escolha a funcionalidade", ["Monitoramento de Estoque", "Detecção em Tempo Real"])

if app_mode == "Monitoramento de Estoque":
    st.header("Estoque em Tempo Real")
    show_stock()
elif app_mode == "Detecção em Tempo Real":
    st.header("Detecção em Tempo Real")
    video_detection()
