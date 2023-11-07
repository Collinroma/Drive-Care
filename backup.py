import os
import cv2
import math
import pygame
import pyttsx3
import numpy as np
from tkinter import *
import mediapipe as mp
import streamlit as st
import mysql.connector
from pygame import mixer
import tkinter.messagebox
from datetime import datetime
from tkinter import *
from PIL import ImageTk, Image  
import sqlite3
from tkinter import messagebox

window = Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.state('zoomed')
window.resizable(15, 15)
window.title('Drive Care Login and Registration Page')

# Window Icon Photo
icon = PhotoImage(file='picco.png')
window.iconphoto(True, icon)

LoginPage = Frame(window)
RegistrationPage = Frame(window)

for frame in (LoginPage, RegistrationPage):
    frame.grid(row=0, column=0, sticky='nsew')


def show_frame(frame):
    frame.tkraise()


show_frame(LoginPage)


# ========== DATABASE VARIABLES ============
Email = StringVar()
FullName = StringVar()
Password = StringVar()
ConfirmPassword = StringVar()

# =====================================================================================================================
# =====================================================================================================================
# ==================== LOGIN PAGE =====================================================================================
# =====================================================================================================================
# =====================================================================================================================

design_frame1 = Listbox(LoginPage, bg='#BDEDFF', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame1.place(x=0, y=0)

design_frame2 = Listbox(LoginPage, bg='#43C6DB', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame2.place(x=676, y=0)

design_frame3 = Listbox(LoginPage, bg='#43C6DB', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame3.place(x=75, y=106)

design_frame4 = Listbox(LoginPage, bg='#f8f8f8', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame4.place(x=676, y=106)

# ====== Email ====================
email_entry = Entry(design_frame4, fg="#151B54", font=("Times New Roman semibold", 12), highlightthickness=2,
                    textvariable=Email)
email_entry.place(x=134, y=170, width=256, height=34)
email_entry.config(highlightbackground="#BDEDFF", highlightcolor="#25383C")
email_label = Label(design_frame4, text='• User Account', fg="#25383C", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
email_label.place(x=130, y=140)

# ==== Password ==================
password_entry1 = Entry(design_frame4, fg="#151B54", font=("Times New Roman semibold", 12), show='•', highlightthickness=2,
                        textvariable=Password)
password_entry1.place(x=134, y=250, width=256, height=34)
password_entry1.config(highlightbackground="#BDEDFF", highlightcolor="black")
password_label = Label(design_frame4, text='• Password', fg="#25383C", bg='#f8f8f8', font=("Times new roman", 11, 'bold'))
password_label.place(x=130, y=220)


# function for show and hide password
def password_command():
    if password_entry1.cget('show') == '•':
        password_entry1.config(show='')
    else:
        password_entry1.config(show='•')


# ====== checkbutton ==============
checkButton = Checkbutton(design_frame4, bg='#f8f8f8', command=password_command, text='show password')
checkButton.place(x=140, y=288)

# ========= Buttons ===============
SignUp_button = Button(LoginPage, text='Sign up', font=("Times New Roman bold", 12), bg='#f8f8f8', fg="#89898b",
                       command=lambda: show_frame(RegistrationPage), borderwidth=0, activebackground='#1b87d2', cursor='hand2')
SignUp_button.place(x=1100, y=175)

# ===== Welcome Label ==============
welcome_label = Label(design_frame4, text='Welcome To Drive-Care', font=('Times New Roman', 20, 'bold'), bg='#f8f8f8')
welcome_label.place(x=130, y=15)

# ======= top Login Button =========
login_button = Button(LoginPage, text='Login', font=("Times New Roman bold", 12), bg='#f8f8f8', fg="#89898b",
                      borderwidth=0, activebackground='#1b87d2', cursor='hand2')
login_button.place(x=845, y=175)

login_line = Canvas(LoginPage, width=60, height=5, bg='#1b87d2')
login_line.place(x=840, y=203)

# ==== LOGIN  down button ============
loginBtn1 = Button(design_frame4, fg='#f8f8f8', text='Login', bg='#43C6DB', font=("Times New Roman bold", 15),
                   cursor='hand2', activebackground='#1b87d2', command=lambda: login())
loginBtn1.place(x=133, y=340, width=256, height=50)


# ===== Email icon =========
email_icon = Image.open('images\\pic-icon.png')
photo = ImageTk.PhotoImage(email_icon)
emailIcon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
emailIcon_label.image = photo
emailIcon_label.place(x=65, y=160)

# ===== password icon =========
password_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(password_icon)
password_icon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
password_icon_label.image = photo
password_icon_label.place(x=105, y=254)

# ===== picture icon =========
picture_icon = Image.open('images\\pic-icon.png')
photo = ImageTk.PhotoImage(picture_icon)
picture_icon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
picture_icon_label.image = photo
picture_icon_label.place(x=420, y=5)

# ===== Left Side Picture ============
side_image = Image.open('picco.png')
photo = ImageTk.PhotoImage(side_image)
side_image_label = Label(design_frame3, image=photo, bg='#1e85d0')
side_image_label.image = photo
side_image_label.place(x=0, y=0)


# ============ LOGIN DATABASE CONNECTION =========
connection = sqlite3.connect('RegLog.db')
cur = connection.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS RegLog(Email TEXT PRIMARY KEY, FullName TEXT, Password TEXT, "
            "ConfirmPassword TEXT)")
connection.commit()
connection.close()


def login():
    connection = sqlite3.connect("RegLog.db")
    cursor = connection.cursor()

    find_user = 'SELECT * FROM RegLog WHERE Email = ? and Password = ?'
    cursor.execute(find_user, [(email_entry.get()), (password_entry1.get())])

    result = cursor.fetchall()
    
    if result:
        messagebox.showinfo("Success", 'Logged in Successfully.')
        window.destroy()
    else:
        messagebox.showerror("Failed", "Wrong Login details, please try again.")
        window.mainloop()

    
    root = Tk()
    height = 500
    width = 450
    x = (root.winfo_screenwidth()//2)-(width//2)
    y = (root.winfo_screenheight()//2)-(height//2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    #root.geometry('500x570')
    frame = Frame(root, relief=RIDGE, borderwidth=2)
    frame.pack(fill=BOTH, expand=1)
    root.title('Drive Main')
    frame.config(background='#74B0D6')
    label = Label(frame, text="Drive Care", bg='#74B0D6', font='Garcedo 25 bold')
    label.pack(side=TOP)
    filename = PhotoImage(file="picco.png")
    background_label = Label(frame, image=filename)
    background_label.pack(side=TOP)


    def hel_doc():
        help(cv2)


    def Contri():
        tkinter.messagebox.showinfo("Contributors", "\n1.Romairo Reid\n2. Kayla-Marie Sooman \n3. Bradly Walcott \n4. Lee Hinds \n")


    def anotherWin():
        tkinter.messagebox.showinfo("About",
                                    'Drive Care version 01.1\n Made Using\n-OpenCV\np\n-Tkinter\n In Python 3')


    menu = Menu(root)
    root.config(menu=menu)

    subm1 = Menu(menu)
    menu.add_cascade(label="Tools", menu=subm1)
    subm1.add_command(label="Open CV Docs", command=hel_doc)

    subm2 = Menu(menu)
    menu.add_cascade(label="About", menu=subm2)
    subm2.add_command(label="Driver Cam", command=anotherWin)
    subm2.add_command(label="Contributors", command=Contri)



    def exitt():
        exit()

    def web():
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break
        capture.release()
        cv2.destroyAllWindows()
  

    def testbutton():
        mixer.init()
        voice_left = mixer.Sound('left.wav')
        voice_right = mixer.Sound('Right.wav')
        voice_down = mixer.Sound('down.wav')
        eyes_blink= mixer.Sound('eyes_blink.wav')
        yawn = mixer.Sound('Yawning.wav')

        counter_right=0
        counter_down=0
        counter_left=0
        FONTS =cv2.FONT_HERSHEY_COMPLEX
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
        BLACK = (0,0,0)
        WHITE = (255,255,255)
        LIGHTBLUE = (180,130,70)
        BLUE = (255,0,0)
        RED = (0,0,255)
        CYAN = (255,255,0)
        YELLOW =(0,255,255)
        MAGENTA = (255,0,255)
        GRAY = (128,128,128)
        GREEN = (0,255,0)
        PURPLE = (128,0,128)
        ORANGE = (0,165,255)
        PINK = (147,20,255)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)

        def landmarksDetection(img, results, draw=False):
            img_height, img_width= img.shape[:2]
            mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
            if draw :
                [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
            return mesh_coord

        def euclaideanDistance(point, point1):
            x, y = point
            x1, y1 = point1
            distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
            return distance

        def blinkRatio(img, landmarks, right_indices, left_indices):
            # Right eyes 
            # horizontal line 
            rh_right = landmarks[right_indices[0]]
            rh_left = landmarks[right_indices[8]]
            # vertical line 
            rv_top = landmarks[right_indices[12]]
            rv_bottom = landmarks[right_indices[4]]
            # draw lines on right eyes 
            # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
            # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

            # LEFT_EYE 
            # horizontal line 
            lh_right = landmarks[left_indices[0]]
            lh_left = landmarks[left_indices[8]]

            # vertical line 
            lv_top = landmarks[left_indices[12]]
            lv_bottom = landmarks[left_indices[4]]

            rhDistance = euclaideanDistance(rh_right, rh_left)
            rvDistance = euclaideanDistance(rv_top, rv_bottom)

            lvDistance = euclaideanDistance(lv_top, lv_bottom)
            lhDistance = euclaideanDistance(lh_right, lh_left)
            
            if lvDistance != 0 and lhDistance !=0:
                reRatio = rhDistance/rvDistance
                leRatio = lhDistance/lvDistance
            
            ratio = (reRatio+leRatio)/2
            return ratio 


        def MouthRatio(img, landmarks, top_indices, bottom_indices):

            lip_right = landmarks[bottom_indices[0]]
            lip_left = landmarks[bottom_indices[10]]

            lip_top = landmarks[top_indices[4]]
            lip_bottom = landmarks[bottom_indices[5]]

            
            lipDistance = euclaideanDistance(lip_top, lip_bottom)

            return lipDistance 


        def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
          
            (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
            x, y = textPos
            cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
            cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

            return img


        Threshold_Frame = [200,350,450]
        counter = 0
        counter_eye = 0
        counter_mouth = 0 
        Counter_right=0
        Counter_down=0
        Counter_left=0
        counter_left=0
        counter_right=0
        counter_down=0

        while cap.isOpened():
            success, image = cap.read()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = face_mesh.process(image)
            
            # To improve performance
            image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
           
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])       
                    
                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360


                    if y< -10:
                        text = "Looking Left"
                        Counter_left += 1             

                    if y > 10:
                        text = "Looking Right"
                        Counter_right += 1

                    if x < -4:
                        text = "Looking Down"
                        Counter_down += 1
                      
                    else:
                        text = "Looking Forward"
                        
                    if y< -10:
                        Counter_right=0
                        Counter_down=0
                        Counter_forward=0
                        counter_down=0
                        counter_right=0
                        if Counter_left % Threshold_Frame[counter_left] == 0  and pygame.mixer.get_busy()==0:
                            counter_left +=1
                            counter_left = counter_left % 3
                            if counter_left == 0:
                                Counter_left = 0
                            
                            voice_left.play()
                    
                    if  y > 10:
                        Counter_left=0
                        Counter_down=0
                        Counter_forward=0
                        counter_left=0
                        counter_down=0
                        if Counter_right > Threshold_Frame[counter_right] and pygame.mixer.get_busy()==0:
                            
                            counter_right +=1
                            counter_right = counter_right % 3  
                            
                            if counter_right == 0:
                                Counter_right = 0
                            
                            voice_right.play()
                            
               
                    if x < -4:
                        Counter_right=0
                        Counter_left=0
                        Counter_forward=0
                        if Counter_down % Threshold_Frame[counter_down] == 0 and pygame.mixer.get_busy()==0:

                            counter_down +=1
                            counter_down = counter_down % 3  
                            
                            if counter_down == 0:
                                Counter_down = 0                    
                            voice_down.play()
                      
                    
                    frame_height, frame_width= image.shape[:2] 
                    
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(image, results, False)
                        ratio = blinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        Mouth_dist= MouthRatio(image, mesh_coords, UPPER_LIPS, LOWER_LIPS)
                        colorBackgroundText(image,  f'Eyes Clsoed for: {counter_eye} frames', FONTS, 0.6, (10,30),2, LIGHTBLUE, WHITE)
                        colorBackgroundText(image,  f'Mouth Open for: {counter_mouth} frames', FONTS, 0.6, (350,30),2, LIGHTBLUE, WHITE)
                        colorBackgroundText(image,  f'Seeing left for: {Counter_left} frames', FONTS, 0.6, (10,60),2, LIGHTBLUE, WHITE)
                        colorBackgroundText(image,  f'Seeing right for: {Counter_right} frames', FONTS, 0.6, (350,60),2, LIGHTBLUE, WHITE)
                        colorBackgroundText(image,  f'Seeing Down for : {Counter_down} frames', FONTS, 0.6, (10,90),2, LIGHTBLUE, WHITE)

                        if ratio > 4.0:
                            counter_eye += 1
                            if counter_eye > 30 and pygame.mixer.get_busy()==0:
                                eyes_blink.play()
                                counter_eye = 0
                        else: 
                            counter_eye=0
                        if 45 < Mouth_dist:
                            counter_mouth += 1
                            if counter_mouth > 50 and pygame.mixer.get_busy()==0:
                                yawn.play()

                                counter_mouth = 0
                        else: 
                            counter_mouth=0
            colorBackgroundText(image, 'Press Button Q to Quit', FONTS, 0.7, (200,460),2, LIGHTBLUE, WHITE)       
            cv2.imshow('Driver Alertness Estimation', image)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break    

        cap.release()
        cv2.destroyAllWindows()


    def testbutton2():
        capture = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        eye_glass = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        op = cv2.VideoWriter('Sample2.avi', fourcc, 9.0, (640, 480))

        while True:
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, 'Face', (x + w, y + h), font, 1, (250, 250, 250), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eye_g = eye_glass.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eye_g:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            op.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xff == ord('e'):
                break
        op.release()
        capture.release()
        cv2.destroyAllWindows()




    but1 = Button(frame, padx=5, pady=5, width=30, bg='white', fg='black', relief=GROOVE, command=web, text='Click to Open Camera',
                  font=('helvetica 14 bold'))
    but1.place(x=5, y=104)

    but2 = Button(frame, padx=5, pady=5, width=30, bg='white', fg='black', relief=GROOVE, command=testbutton, text='Drive-Care (Protection)',
                  font=('helvetica 14 bold'))
    but2.place(x=5, y=176)

    but3 = Button(frame, padx=5, pady=5, width=30, bg='white', fg='black', relief=GROOVE, command=testbutton2, text='Dash Cam',
                  font=('helvetica 14 bold'))
    but3.place(x=5, y=250)

    but5 = Button(frame, padx=4, pady=5, width=5, bg='red', fg='black', relief=GROOVE, text='EXIT', command=exitt,
                  font=('helvetica 14 bold'))
    but5.place(x=153, y=400)

    root.mainloop()
# ===================================================================================================================
# ===================================================================================================================
# === FORGOT PASSWORD  PAGE =========================================================================================
# ===================================================================================================================
# ===================================================================================================================


def forgot_password():
    win = Toplevel()
    window_width = 350
    window_height = 350
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    position_top = int(screen_height / 4 - window_height / 4)
    position_right = int(screen_width / 2 - window_width / 2)
    win.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    win.title('Forgot Password')
    win.iconbitmap('images\\aa.ico')
    win.configure(background='#f8f8f8')
    win.resizable(0, 0)

    # Variables
    email = StringVar()
    password = StringVar()
    confirmPassword = StringVar()

    # ====== Email ====================
    email_entry2 = Entry(win, fg="#151B54", font=("Times new roman semibold", 12), highlightthickness=2,
                         textvariable=email)
    email_entry2.place(x=40, y=30, width=256, height=34)
    email_entry2.config(highlightbackground="#BDEDFF", highlightcolor="black")
    email_label2 = Label(win, text='• Email account', fg="#25383C", bg='#f8f8f8',
                         font=("Times New Roman", 11, 'bold'))
    email_label2.place(x=40, y=0)

    # ====  New Password ==================
    new_password_entry = Entry(win, fg="#151B54", font=("Times new roman semibold", 12), show='•', highlightthickness=2,
                               textvariable=password)
    new_password_entry.place(x=40, y=110, width=256, height=34)
    new_password_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
    new_password_label = Label(win, text='• New Password', fg="#25383C", bg='#f8f8f8', font=("Times new roman", 11, 'bold'))
    new_password_label.place(x=40, y=80)

       # function for show and hide password
    def password_command2():
        if new_password_entry.cget('show') == '•':
            new_password_entry.config(show='')
        else:
            new_password_entry.config(show='•')

            
    checkButton = Checkbutton(win, bg='#f8f8f8', command=password_command2, text='show password')
    checkButton.place(x=21, y=143)
    
    # ====  Confirm Password ==================
    confirm_password_entry = Entry(win, fg="#151B54", font=("Times new roman semibold", 12), show='•', highlightthickness=2
                                   , textvariable=confirmPassword)
    confirm_password_entry.place(x=40, y=190, width=256, height=34)
    confirm_password_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
    confirm_password_label = Label(win, text='• Confirm Password', fg="#25383C", bg='#f8f8f8',
                                   font=("Times new roman", 11, 'bold'))
    confirm_password_label.place(x=40, y=160)
    
    def password_command3():
        if confirm_password_entry.cget('show') == '•':
            confirm_password_entry.config(show='')
        else:
            confirm_password_entry.config(show='•')
            
    checkButton = Checkbutton(win, bg='#f8f8f8', command=password_command3, text='show password')
    checkButton.place(x=20, y=225)     # function for show and hide password
   
    # ======= Update password Button ============
    update_pass = Button(win, fg='#151B54', text='Update Password', bg='#1b87d2', font=("Times new roman bold", 14),
                         cursor='hand2', activebackground='#1b87d2', command=lambda: change_password())
    update_pass.place(x=40, y=270, width=256, height=50)

    # ========= DATABASE CONNECTION FOR FORGOT PASSWORD=====================
    def change_password():
        if new_password_entry.get() == confirm_password_entry.get():
            db = sqlite3.connect("RegLog.db")
            curs = db.cursor()

            insert = '''update RegLog set Password=?, ConfirmPassword=? where Email=? '''
            curs.execute(insert, [new_password_entry.get(), confirm_password_entry.get(), email_entry2.get(), ])
            db.commit()
            db.close()
            messagebox.showinfo('Congrats', 'Password changed successfully')

        else:
            messagebox.showerror('Error!', "Passwords didn't match")


forgotPassword = Button(design_frame4, text='Forgot password', font=("Times new roman", 8, "bold underline"), bg='#f8f8f8',
                        borderwidth=0, activebackground='#f8f8f8', command=lambda: forgot_password(), cursor="hand2")
forgotPassword.place(x=290, y=290)



# =====================================================================================================================
# =====================================================================================================================
# ==================== REGISTRATION PAGE ==============================================================================
# =====================================================================================================================
# =====================================================================================================================

design_frame5 = Listbox(RegistrationPage, bg='#BDEDFF', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame5.place(x=0, y=0)

design_frame6 = Listbox(RegistrationPage, bg='#43C6DB', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame6.place(x=676, y=0)

design_frame7 = Listbox(RegistrationPage, bg='#43C6DB', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame7.place(x=75, y=106)

design_frame8 = Listbox(RegistrationPage, bg='#f8f8f8', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame8.place(x=676, y=106)

# ==== Full Name =======
name_entry = Entry(design_frame8, fg="#151B54", font=("Times new roman semibold", 12), highlightthickness=2,
                   textvariable=FullName)
name_entry.place(x=284, y=150, width=286, height=34)
name_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
name_label = Label(design_frame8, text='•Full Name', fg="#25383C", bg='#f8f8f8', font=("Times new roman", 11, 'bold'))
name_label.place(x=280, y=120)

# ======= Email ===========
email_entry = Entry(design_frame8, fg="#151B54", font=("Times new roman semibold", 12), highlightthickness=2,
                    textvariable=Email)
email_entry.place(x=284, y=220, width=286, height=34)
email_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
email_label = Label(design_frame8, text='•Email', fg="#25383C", bg='#f8f8f8', font=("Times new romans", 11, 'bold'))
email_label.place(x=280, y=190)

# ====== Password =========
password_entry = Entry(design_frame8, fg="#151B54", font=("Times new roman semibold", 12), show='•', highlightthickness=2,
                       textvariable=Password)
password_entry.place(x=284, y=295, width=286, height=34)
password_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
password_label = Label(design_frame8, text='• Password', fg="#25383C", bg='#f8f8f8',
                       font=("Times new romans", 11, 'bold'))
password_label.place(x=280, y=265)


def password_command2():
    if password_entry.cget('show') == '•':
        password_entry.config(show='')
    else:
        password_entry.config(show='•')


checkButton = Checkbutton(design_frame8, bg='#f8f8f8', command=password_command2, text='show password')
checkButton.place(x=290, y=330)


# ====== Confirm Password =============
confirmPassword_entry = Entry(design_frame8, fg="#151B54", font=("Times new roman", 12), highlightthickness=2,
                              textvariable=ConfirmPassword)
confirmPassword_entry.place(x=284, y=385, width=286, height=34)
confirmPassword_entry.config(highlightbackground="#BDEDFF", highlightcolor="black")
confirmPassword_label = Label(design_frame8, text='• Confirm Password', fg="#89898b", bg='#f8f8f8',
                              font=("Times new roman", 11, 'bold'))
confirmPassword_label.place(x=280, y=355)

# ========= Buttons ====================
SignUp_button = Button(RegistrationPage, text='Sign up', font=("Times new roman", 12), bg='#f8f8f8', fg="#89898b",
                       command=lambda: show_frame(LoginPage), borderwidth=0, activebackground='#1b87d2', cursor='hand2')
SignUp_button.place(x=1100, y=175)

SignUp_line = Canvas(RegistrationPage, width=60, height=5, bg='#1b87d2')
SignUp_line.place(x=1100, y=203)

# ===== Welcome Label ==================
welcome_label = Label(design_frame8, text='Welcome To Drive-Care', font=('Times new roman', 20, 'bold'), bg='#f8f8f8')
welcome_label.place(x=130, y=15)

# ========= Login Button =========
login_button = Button(RegistrationPage, text='Login', font=("Times new roman bold", 12), bg='#f8f8f8', fg="#89898b",
                      borderwidth=0, activebackground='#1b87d2', command=lambda: show_frame(LoginPage), cursor='hand2')
login_button.place(x=845, y=175)

# ==== SIGN UP down button ============
signUp2 = Button(design_frame8, fg='#f8f8f8', text='Sign Up', bg='#43C6DB', font=("Times new roman bold", 15),
                 cursor='hand2', activebackground='#1b87d2', command=lambda: submit())
signUp2.place(x=285, y=435, width=286, height=50)

# ===== password icon =========
password_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(password_icon)
password_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
password_icon_label.image = photo
password_icon_label.place(x=255, y=300)

# ===== confirm password icon =========
confirmPassword_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(confirmPassword_icon)
confirmPassword_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
confirmPassword_icon_label.image = photo
confirmPassword_icon_label.place(x=255, y=390)

# ===== Email icon =========
email_icon = Image.open('images\\email-icon.png')
photo = ImageTk.PhotoImage(email_icon)
emailIcon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
emailIcon_label.image = photo
emailIcon_label.place(x=255, y=225)

# ===== Full Name icon =========
name_icon = Image.open('images\\name-icon.png')
photo = ImageTk.PhotoImage(name_icon)
nameIcon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
nameIcon_label.image = photo
nameIcon_label.place(x=252, y=153)

# ===== picture icon =========
picture_icon = Image.open('images\\pic-icon.png')
photo = ImageTk.PhotoImage(picture_icon)
picture_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
picture_icon_label.image = photo
picture_icon_label.place(x=420, y=5)

# ===== Left Side Picture ============
side_image = Image.open('picco.png')
photo = ImageTk.PhotoImage(side_image)
side_image_label = Label(design_frame7, image=photo, bg='#1e85d0')
side_image_label.image = photo
side_image_label.place(x=0, y=0)


# =====================================================================================================================
# =====================================================================================================================
# ==================== DATABASE CONNECTION ============================================================================
# =====================================================================================================================
# =====================================================================================================================

connection = sqlite3.connect('RegLog.db')
cur = connection.cursor() #get me this data find me what i want
cur.execute("CREATE TABLE IF NOT EXISTS RegLog(Email TEXT PRIMARY KEY, FullName TEXT, Password TEXT, "
            "ConfirmPassword TEXT)")#create data base through cur
connection.commit()
connection.close()


def submit():
    check_counter = 0
    warn = ""
    if name_entry.get() == "":
        warn = "Missing Information"
    else:
        check_counter += 1

    if email_entry.get() == "":
        warn = "Email Field Missing Information"
    else:
        check_counter += 1

    if password_entry.get() == "":
        warn = "Password Missing Information"
    else:
        check_counter += 1

    if confirmPassword_entry.get() == "":
        warn = "Unable to sign up,please to enter required information and try again"
    else:
        check_counter += 1

    if password_entry.get() != confirmPassword_entry.get():
        warn = "Confirm Passwords no match!"
    else:
        check_counter += 1

    if check_counter == 5:
        try:
            connection = sqlite3.connect("RegLog.db")
            cur = connection.cursor()
            cur.execute("INSERT INTO RegLog values(?,?,?,?)",
                        (Email.get(), FullName.get(), Password.get(), ConfirmPassword.get()))

            connection.commit()
            connection.close()
            messagebox.showinfo("Successful Registration", "Welcome to Drive-Care")

        except Exception as ep:
            messagebox.showerror('', ep)
    else:
        messagebox.showerror('Error', warn)


window.mainloop()

