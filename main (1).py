import sys
import argparse
import numpy as np
import cv2 as cv
import easygui
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import heartpy as hp
from heartpy.datautils import rolling_mean, _sliding_window
import pandas as pd




def main(caminho=None):  # variavel vazia
    """Lê o vídeo a partir do arquivo ou da webcam."""
    try:
        if caminho is None:
            cap = cv.VideoCapture(0)  # ler a partir da webcam
            if not cap.isOpened():
                raise IOError
        else:
            cap = cv.VideoCapture(caminho)  # ler a partir do caminho
            if not cap.isOpened():
                raise NameError
    except IOError:
        print("Impossível abrir a webcam, verifique a conexão.")
        sys.exit()
    except NameError:
        print("Erro na leitura, você checou se é um arquivo de vídeo?")
        sys.exit()


    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # numero total de quadros
    fps = cap.get(cv.CAP_PROP_FPS)  # numero de quadros por segundo do vídeo
    raw_ppg = np.zeros([n_frames])  # ppg bruto
    i_frame = 0  # contador para os quadros
    while True:
        frame = cap.read()[1]  # ler o quadro da imagem do vídeo
        if frame is None:  # fim do vídeo
            break
        roi_gray, roi_color = detecta_face(frame)
        if roi_gray is not None:  # se a face foi detectada
            roi_testa = detecta_olho(roi_gray, roi_color)
        else:
            roi_testa = None
        if roi_testa is not None:  # se encontrou a região da testa
            raw_ppg[i_frame] = calcular_media_matiz(roi_testa)
            i_frame += 1
        cv.imshow('Video', frame)  # mostra a imagem capturada na janela

        # o trecho seguinte e apenas para parar o codigo e fechar a janela
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    raw_ppg = raw_ppg[:i_frame]
    fs_real = len(raw_ppg) / fps
    calcular_fc(raw_ppg, fs_real)
    print("Done!")


def detecta_face(frame):
    """Detecta a face no quadro."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade_name = 'haarcascade_frontalface_default.xml'  # classificador
    face_cascade = cv.CascadeClassifier(face_cascade_name)     # para a face
    faces = face_cascade.detectMultiScale(gray, minNeighbors=20,
                                          minSize=(30, 30),
                                          maxSize=(300, 300))
    for (x_coord, y_coord, width, height) in faces:
        roi_gray = gray[y_coord : y_coord+height, x_coord : x_coord+width]
        roi_color = frame[y_coord : y_coord+height, x_coord : x_coord+width]
        cv.rectangle(frame, (x_coord, y_coord),  # retangulo da face
                     (x_coord + width, y_coord + height), (0, 255, 0), 4)
    if len(faces) == 0:
        return None, None
    else:
        return roi_gray, roi_color


def detecta_olho(roi_gray, roi_color):
    """Detecta o olho no quadro."""
    eye_cascade_name = 'haarcascade_eye.xml'              # classificador
    eye_cascade = cv.CascadeClassifier(eye_cascade_name)  # para os olhos
    olhos = eye_cascade.detectMultiScale(roi_gray, minNeighbors=20,
                                         minSize=(10, 10), maxSize=(90, 90))
    contador = 0  # conta a quantidade de olhos encontrados na face
    eye_x = None  # coordenada x de um dos olhos
    eye_y = None  # coordenada y de um dos olhos
    eye_wd = None  # largura de um dos olhos
    eye_hg = None  # altura de um dos olhos
    eyes_wd = None  # largura de ambos os olhos
    eyes_hg = None  # altura de ambos os olhos

    for (eye_x, eye_y, eye_wd, eye_hg) in olhos:
        if contador == 0:  # armazena os dados de um olho antes do próximo
            eye_x_temp = eye_x
            eye_y_temp = eye_y
            eye_width_temp = eye_wd
            eye_height_temp = eye_hg
            eye_height_bottom = eye_y_temp + eye_height_temp
        contador += 1

    if contador == 2:  # quando os dois olhos são encontrados no quadro
        if eye_x_temp < eye_x:
            eyes_wd = eye_x + eye_wd - eye_x_temp
            eye_x = eye_x_temp
        else:
            eyes_wd = eye_x_temp + eye_width_temp - eye_x
        if eye_height_bottom < eye_y + eye_hg:
            eye_height_bottom = eye_y + eye_hg
        if eye_y_temp < eye_y:
            eyes_hg = eye_height_bottom - eye_y_temp
            eye_y = eye_y_temp
        else:
            eyes_hg = eye_height_bottom - eye_y
    elif contador == 1:  # quando só um olho é encontrado no quadro
        width = roi_gray.shape[1]  # largura do quadro
        eyes_hg = eye_hg
        ponto_medio = width / 2  # ponto medio da largura da face
        if eye_x > ponto_medio:  # caso encontre só o olho direito do quadro
            eye_x = width - (eye_x + eye_wd)
        eyes_wd = width - 2*eye_x

    if contador == 0 or len(olhos) > 2:  # nenhum olho encontrado no quadro
        roi_testa = None
    else:
        testa_x = eye_x + int(0.5*eye_wd)
        testa_y = 0
        testa_w = eyes_wd - eye_wd
        testa_h = int(0.7*eyes_hg)
        cv.rectangle(roi_color, (eye_x, eye_y),  # retangulo dos olhos
                     (eye_x + eyes_wd, eye_y + eyes_hg), (255, 0, 0), 2)
        cv.rectangle(roi_color, (testa_x, testa_y),  # retângulo da testa
                     (testa_x + testa_w, testa_y + testa_h), (255, 0, 255), 2)
        roi_testa = roi_color[testa_y : testa_y+testa_h,
                              testa_x : testa_x+testa_w]
    return roi_testa


def gravar_video(nome):
    """Grava o vídeo a partir da webcam."""
    cap = cv.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    arquivo = "testevideo_" + nome + ".avi"
    out = cv.VideoWriter(arquivo, fourcc, 30.0, (640, 480))  # 30fps
    contador = 0  # contador do numero de quadros
    for __ in range(900+150):  # 900 (duracao do video)+150 (5 segundos)
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if contador > 150:
            out.write(frame)
        # moldura e barras de progresso
        cv.rectangle(frame, (200, 120), (440, 400), (0, 0, 0), 2)

        if contador < 180+150:  # primeira barra
            cv.rectangle(frame, (70-10, 440), (70+10, 480), (0, 255, 0), -1)
        if contador < (180*2) + 150:  # segunda barra
            cv.rectangle(frame, (170-10, 440), (170+10, 480), (0, 255, 0), -1)
        if contador < (540+150):  # terceira barra
            cv.rectangle(frame, (270-10, 440), (270+10, 480), (0, 255, 0), -1)
        if contador < (720+150):  # quarta barra
            cv.rectangle(frame, (370-10, 440), (370+10, 480), (0, 255, 0), -1)
        if contador < (900+150):  # quinta barra
            cv.rectangle(frame, (470-10, 440), (470+10, 480), (0, 255, 0), -1)
        contador += 1
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


def calcular_media_matiz(roi_testa):
    """Calcula a média de matiz da região da testa."""
    hsv = cv.cvtColor(roi_testa, cv.COLOR_BGR2HSV)  # Converte BGR em HSV
    try:
        media_matiz = np.average(hsv[:, :, 0], weights=(hsv[:, :, 0]) < 18)
    except ZeroDivisionError:
        media_matiz = 0
    return media_matiz


def calcular_fc(raw_ppg, fs):
    """Calcula a frequência cardíaca a partir do sinal PPG bruto."""
    # calcula frequencia cardiaca- IIR BAND PASS BUTHERWORTH
    T = 1/fs  # periodo de amostragem
    nyq = 0.5 * fs  # frequência de Nyquist
    freq_a = 0.8 / nyq  # corte de limite inferior para filtro passa-faixa
    freq_b = 2.2 / nyq  # corte de limite superior para filtro passa-faixa

    b, a = butter(2, (freq_a, freq_b), btype='bandpass')  # filtro de ordem 2
    ppg_filtrado = filtfilt(b, a, raw_ppg)  # obtem o sinal filtrado

    # calculo fft do sinal filtrado
    N = ppg_filtrado.size
    t = np.linspace(0, N * T, N)

    fft = np.fft.fft(ppg_filtrado)

    # fornece os componentes de frequência correspondentes aos dados
    freq = np.fft.fftfreq(len(ppg_filtrado), T)
    frequencia = freq[:N // 2]
    amplitude = np.abs(fft)[:N // 2] * 1 / N  # normalizando

    indice_max = np.argmax(amplitude)
    freq_max = frequencia[indice_max]

    # sinal do csv
    df = pd.read_csv(PATH_CSV, usecols=["Name", "Marks"])


    #RMSSD 
    #ppg_filtrado, timer = hp.load_exampledata(0)
    wd, m = hp.process(ppg_filtrado, sample_rate = fs)
    wd['peaklist'] #mostra a lista de picos 
    figura = hp.plotter(wd, m, show=False, title='Picos do sinal FPG')
    figura.savefig("picos.png", dpi=600)
    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2)
    
    # For Sine Function
    axis[0, 0].plot(t, raw_ppg)
    axis[0, 0].set_title("sinal bruto de amplitude no tempo")
    axis[0, 0].set_ylabel("Amplitude")
    axis[0, 0].set_xlabel("Time [s]")

    # For Cosine Function
    axis[0, 1].plot(t, ppg_filtrado)
    axis[0, 1].set_title("sinal filtrado de amplitude no tempo")
    
    # For Tangent Function
    axis[1, 0].plot(t, ppg_filtrado)
    axis[1, 0].set_title("sinal fft de amplitude em frequência")
    
    # For Tanh Function
    axis[1, 1].plot(df.Name, df.Marks)
    axis[1, 1].set_title("sinal OCG")
    
    # Combine all the operations and display
    plt.show()
    


if __name__ == "__main__":
    print('============== Bem vindo ao Análisador de Photopletismografia ==============')
    print('Selecione a opção que desejar:')
    print('1 - Gravar pela Webcan')
    print('2 - Selecionar vídeo no computador')
    record = int(input('resposta: '))
    if record==1:
        gravar_video(input("Digite o nome para o video: "))
    else:
        print('Por favor selecione seu arquivo de video')
        path_video = easygui.fileopenbox()
        print('Agora selecione seu o arquivo "csv" com os dados do OCG')
        PATH_CSV = easygui.fileopenbox(filetypes=['*.csv'])
        main(path_video)  # escolhe o video nas pastas locais
