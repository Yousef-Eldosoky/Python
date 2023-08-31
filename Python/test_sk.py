import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import librosa


def audio_recognition(path_of_audio):
    data = pd.read_csv("/Users/yousefadel/Downloads/training data.csv")

    # we need to let Male = 1 and female = 0 for further data engineering
    data['label'] = data['label'].replace('male', 1)
    data['label'] = data['label'].replace('female', 0)

    X = data[['sd', 'Q25', 'IQR', 'mode', 'meanfun']].values
    Y = data['label'].values

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearSVC()
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    y_preds = model.predict(x_test)
    # print(f"Training score = {metrics.accuracy_score(y_train, train_pred)}")
    # print(f"Testing score = {metrics.accuracy_score(y_test, y_preds)}")

    # Load the audio file
    y, sr = librosa.load(path_of_audio)

    # Compute the parameters
    sd = np.std(librosa.fft_frequencies(sr=sr, n_fft=2048))
    Q25 = np.quantile(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)), 0.25)
    IQR = np.quantile(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)), 0.75) - Q25
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hist, bin_edges = np.histogram(
        np.round(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2))), bins=np.arange(129))
    mode_bin = np.argmax(hist)
    mode = librosa.midi_to_hz(mode_bin + 1)
    Fun = librosa.feature.rms(y=y)
    meanfun = Fun.mean()
    # minfun = np.min(Fun)
    # maxfun = np.max(Fun)

    input_data = [sd / 10000, librosa.midi_to_hz(Q25) / 1000, librosa.midi_to_hz(IQR) / 1000, mode / 10000, meanfun]

    # saving the model
    filename_LinearSVC = 'trained_model.sav'
    pickle.dump(model, open(filename_LinearSVC, 'wb'))
    # loading the saved model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        Prediction = 'The person is female'
    else:
        Prediction = 'The person is male'

    return Prediction


# pred = audio_recognition("/Users/yousefadel/Downloads/ecoute_moi.wav")
# print(pred)

import cv2
import sys
from PIL import Image
from rembg import remove
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def remove_background(inputpath):
    outputpath = 'background_removed.png'
    input = Image.open(inputpath)
    output = remove(input)
    output.save(outputpath)
    return output


def image_treatment(imagePath):
    image = cv2.imread(imagePath)

    # convert image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,  # number of neighbor faces (effect on the accuracy)
                                         minSize=(30, 30)
                                         )

    # print("[INFO] Found {0} Faces!".format(len(faces)))

    # print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    status = cv2.imwrite('faces_detected.jpg', image)
    # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    img = Image.open(imagePath)
    box = (x, y, x + w, y + h)
    cropped_img = img.crop(box)
    Background_removed = cropped_img.save('Background_removed.jpg')
    return cropped_img


def PCA_analysis(image):
    # this function should take the gray cropped image and apply pca to minimize the features
    img_arr = np.asarray(image)
    img_arr_reshaped = img_arr.reshape(-1, img_arr.shape[-1])
    pca = PCA(n_components=75)
    pca.fit(img_arr_reshaped)
    projected_data = pca.transform(img_arr_reshaped)
    reconstructed_data = pca.inverse_transform(projected_data)
    reconstructed_img = np.reshape(reconstructed_data, img_arr.shape)
    return reconstructed_img


# Euclidean Distance another way
def euclidean_distance(img1_vec, img2_vec):
    distance = np.linalg.norm(img1_vec - img2_vec)
    return distance


def recognise(path1, path2, path3, path4):
    front_view = image_treatment(path1).convert("L")
    left_view = remove_background(path2).convert("L")
    right_view = remove_background(path3).convert("L")
    test_img = image_treatment(path4).convert("L")
    front_view_pca = PCA_analysis(front_view)
    left_view_pca = PCA_analysis(left_view)
    right_view_pca = PCA_analysis(right_view)
    test_img_pca = PCA_analysis(test_img)
    shape = (400, 400)
    front_view_resized = cv2.resize(front_view_pca, shape)
    left_view_resized = cv2.resize(left_view_pca, shape)
    right_view_resized = cv2.resize(right_view_pca, shape)
    test_img_resized = cv2.resize(test_img_pca, shape)
    front_vec = front_view_resized.flatten()
    left_vec = left_view_resized.flatten()
    right_vec = right_view_resized.flatten()
    test_vec = test_img_resized.flatten()
    pos_dist1 = euclidean_distance(front_vec, test_vec)
    pos_dist2 = euclidean_distance(left_vec, test_vec)
    pos_dist3 = euclidean_distance(right_vec, test_vec)
    distance_list = [pos_dist1, pos_dist2, pos_dist3]
    distance = sum(distance_list) / len(distance_list)
    threshold = 33900
    if (distance <= threshold):
        return "this is the same person"
    else:
        return "this is not the same person"


# recognise('/content/training_img1.jpeg', '/content/training_img2.jpeg', '/content/training_img3.jpeg',
#           '/content/testing_img.jpeg')

import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from tkinter import filedialog
import time
import tkinter.ttk as tkk
import os
import wave
import threading
import pyaudio


class MyGUI:
    def __init__(self):
        self.root = tk.Tk()

        self.root.title('Final Project')

        self.root.geometry("800x500")

        self.label = tk.Label(self.root, text="Knock...Knock.....", font=('Arial', 30))
        self.label.place(x=250, y=120)

        self.label6 = tk.Label(self.root, text="Who's there?", font=('Arial', 30))
        self.label6.place(x=350, y=200)

        # self.check_var = tk.IntVar()
        #
        # self.check = tk.Checkbutton(self.root, text="Knn", font=('Arial', 16), variable=self.check_var)
        # self.check.place(x=60, y=140)

        self.label = tkk.Label(self.root,
                               text="This is the main window")

        self.label.pack(pady=10)

        # a button widget which will open a
        # new window on button click

        self.button = tk.Button(self.root, text="Voice\nRecognition", height=5, width=8, fg='#9754C2',
                                font=('Arial', 16),
                                command=self.voice_rec)

        self.button.place(x=200, y=250)

        self.button1 = tk.Button(self.root, text="Face\nRecognition", height=5, width=8, fg='#9754C2',
                                 font=('Arial', 16),
                                 command=self.image_rec)

        self.button1.place(x=470, y=250)

        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def startAlgo(self):

        if self.x:
            self.path_aud = 'recording1.wav'
            self.x = False

        pred = audio_recognition(self.path_aud)

        # result = 'Male'
        messagebox.showinfo(title='Result', message=pred)

        # print(self.path_aud)

        # print('Hello, World!')

    def startAlgo_f(self):
        pred = recognise(self.path_f, self.path_r, self.path_l, self.path_c)

        # result = 'Male'
        messagebox.showinfo(title='Result', message=pred)

        # print('Hello, World!')

    def open_file(self):
        self.file_path = filedialog.askopenfile(mode='r', title="Select a file",
                                                filetypes=[('Sound file', '*wav'), ('All files', '*.*')])
        if self.file_path is not None:
            pass

        self.file_path = str(self.file_path)

        y = False

        self.path_aud = ''

        i = 0

        for x in self.file_path:
            if x == "'":
                y = True
                i += 1
                continue
            if i == 2:
                break
            if y:
                self.path_aud = self.path_aud + x

    def open_file_f(self):
        self.file_path = filedialog.askopenfile(mode='r', title="Select a file",
                                                filetypes=[('Image file', '*jpeg *jpg'), ('All files', '*.*')])
        if self.file_path is not None:
            pass

        self.file_path = str(self.file_path)

        y = False

        self.path_f = ''

        i = 0

        for x in self.file_path:
            if x == "'":
                y = True
                i += 1
                continue
            if i == 2:
                break
            if y:
                self.path_f = self.path_f + x

    def open_file_r(self):
        self.file_path = filedialog.askopenfile(mode='r', title="Select a file",
                                                filetypes=[('Image file', '*jpeg *jpg'), ('All files', '*.*')])
        if self.file_path is not None:
            pass

        self.file_path = str(self.file_path)

        y = False

        self.path_r = ''

        i = 0

        for x in self.file_path:
            if x == "'":
                y = True
                i += 1
                continue
            if i == 2:
                break
            if y:
                self.path_r = self.path_r + x

    def open_file_l(self):
        self.file_path = filedialog.askopenfile(mode='r', title="Select a file",
                                                filetypes=[('Image file', '*jpeg *jpg'), ('All files', '*.*')])
        if self.file_path is not None:
            pass

        self.file_path = str(self.file_path)

        y = False

        self.path_l = ''

        i = 0

        for x in self.file_path:
            if x == "'":
                y = True
                i += 1
                continue
            if i == 2:
                break
            if y:
                self.path_l = self.path_l + x

    def open_file_c(self):
        self.file_path = filedialog.askopenfile(mode='r', title="Select a file",
                                                filetypes=[('Image file', '*jpeg *jpg'), ('All files', '*.*')])
        if self.file_path is not None:
            pass

        self.file_path = str(self.file_path)

        y = False

        self.path_c = ''

        i = 0

        for x in self.file_path:
            if x == "'":
                y = True
                i += 1
                continue
            if i == 2:
                break
            if y:
                self.path_c = self.path_c + x

    def voice_rec(self):
        self.root = tk.Tk()

        self.root.title('Final Project')

        self.root.geometry("800x500")

        self.label = tk.Label(self.root, text="This program will try to guess your gender.", font=('Arial', 18))
        self.label.place(x=220, y=100)

        self.label1 = tk.Label(self.root, text="Let's see the result!", font=('Arial', 18))
        self.label1.place(x=270, y=150)

        # self.label2 = tk.Label(self.root, text="Choose the algorithm you want to train with:", font=('Arial', 18))
        # self.label2.place(x=4, y=100)

        self.label3 = tk.Label(self.root, text="Upload audio file", font=('Arial', 14))
        self.label3.place(x=30, y=253)

        self.x = False

        # self.check_var = tk.IntVar()
        #
        # self.check = tk.Checkbutton(self.root, text="Knn", font=('Arial', 16), variable=self.check_var)
        # self.check.place(x=60, y=140)

        self.button = tk.Button(self.root, text="Predict", height=3, fg='#9754C2', font=('Arial', 16),
                                command=self.startAlgo)
        self.button.place(x=330, y=350)

        self.button1 = tk.Button(self.root, text="Choose File", height=0, font=('Arial', 16), command=self.open_file)
        self.button1.place(x=150, y=250)

        self.rec_button = tk.Button(self.root, text="🎤-", font=("Arial", 50, "bold"), command=self.recording_voice)
        self.rec_button.place(x=500, y=210)

        self.label4 = tk.Label(self.root, text="00:00:00")
        self.label4.place(x=500, y=280)

        self.recording = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def recording_voice(self):
        self.x = True
        if self.recording:
            self.recording = False
            self.rec_button.config(fg="black")
        else:
            self.recording = True
            self.rec_button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)
        frames = []

        start = time.time()

        while self.recording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label4.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # exists = True
        i = 1

        # while exists:
        #     if os.path.exists(f"recording{i}.wav"):
        #         i += 1
        #     else:
        #         exists = False

        sound_file = wave.open(f"recording{i}.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

    def on_closing(self):
        if messagebox.askyesno(title="Quite?", message="Do you really want to quit?"):
            messagebox.showinfo(title='Quiting', message='Good Bye!')
            # print("Good Bye!")
            self.root.destroy()

    def image_rec(self):
        self.root = tk.Tk()

        self.root.title('Final Project')

        self.root.geometry("800x500")

        self.label = tk.Label(self.root, text="This program will try to guess if you are the same person.",
                              font=('Arial', 18))
        self.label.place(x=180, y=40)

        self.label1 = tk.Label(self.root, text="Let's see the result!", font=('Arial', 20))
        self.label1.place(x=270, y=100)

        # self.label2 = tk.Label(self.root, text="Choose the algorithm you want to train with:", font=('Arial', 18))
        # self.label2.place(x=4, y=100)

        self.label3 = tk.Label(self.root, text="Upload The front image", font=('Arial', 14))
        self.label3.place(x=250, y=145)

        self.label4 = tk.Label(self.root, text="Upload The left side image", font=('Arial', 14))
        self.label4.place(x=250, y=190)

        self.label5 = tk.Label(self.root, text="Upload The right side image", font=('Arial', 14))
        self.label5.place(x=250, y=235)

        self.label6 = tk.Label(self.root, text="Upload The check image", font=('Arial', 16))
        self.label6.place(x=250, y=300)

        # self.check_var = tk.IntVar()
        #
        # self.check = tk.Checkbutton(self.root, text="Knn", font=('Arial', 16), variable=self.check_var)
        # self.check.place(x=60, y=140)

        self.button = tk.Button(self.root, text="Check", height=3, fg='#9754C2', font=('Arial', 16),
                                command=self.startAlgo_f)
        self.button.place(x=330, y=390)

        self.button1 = tk.Button(self.root, text="Choose File", height=0, font=('Arial', 16), command=self.open_file_f)
        self.button1.place(x=450, y=145)

        self.button2 = tk.Button(self.root, text="Choose File", height=0, font=('Arial', 16), command=self.open_file_r)
        self.button2.place(x=450, y=190)

        self.button3 = tk.Button(self.root, text="Choose File", height=0, font=('Arial', 16), command=self.open_file_l)
        self.button3.place(x=450, y=235)

        self.button4 = tk.Button(self.root, text="Choose File", height=0, font=('Arial', 16), command=self.open_file_c)
        self.button4.place(x=450, y=300)

        self.recording = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()


MyGUI()




