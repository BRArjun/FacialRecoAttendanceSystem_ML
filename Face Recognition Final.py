
#STEPS:

#1.Look at the picture and find all the faces in it
#2.Focus on each face and be able to understand it under different conditions
#3.Look at and extract certain unique features in every face to tell each face apart
#4.Compare this feature data with the person to be identified to determine his/her identity



#WORKFLOW


#1.
#We start off by making the image to be analyzed black and white because we will not be needing colour data for finding faces.

#We do this to avoid the confusion of a very bright and a very dark picture of a person not being recognized

#We find all the faces in the image by using an algorithm called HOG(Histogram of Oriented Gradients) which analyzes the basic flow #of light in the image and draws the gradients from the lighter pixels to the darker pixels by dividing the image into 16x16
#squares and checking the overall gradient direction.

#Gradients are simply arrows in the direction of decreasing brightness of the pixels in the image.

#Comparing this HOG image to HOGs of other training faces of the same person we can find the region where the face is located in an image.   



#2.
#For images where the person is situated at a different angle we warp each picture so that the eyes and lips are always
#in the center of the image.

#We use an algorithm called Face Landmark Estimation for this.The idea is we use 68 different points in a face and train a model #that can recognise these 68 features on any face.


#After this step we can rotate or scale the image so that parallel lines are maintained in the image and no other distortions are 
#present.These are called Affine Transformations.



#3.
#To tell faces apart we start with a Deep Convolutional Neural Network that generates 128 measurements (called embedding) of a #face(eg:Distance between eyes,ear length etc.)

#This Network works like this:
#1.It takes 3 images(2 of the same person and 1 of a different person)
#2.Makes sure that the encodings it generates are almost similar for 2 pictures of the same person and different for the other by 
#tweaking it slightly.  




#4.
#To finally output the name of the person we have a simple SVM classifier model that can take in all the 128 measurements of the #person compare the 128 measurements of the face to every other known face and return the image with the lowest face distance.






 
import face_recognition
import numpy as np
from datetime import datetime
import os
import cv2
import keyboard


#FOR THE GUI
import pyautogui
import customtkinter as cstk
import tkinter as tk
from tkinter import filedialog, PhotoImage


cstk.set_appearance_mode("dark")
cstk.set_default_color_theme("green")
root = cstk.CTk()
root.geometry("1920x1080")
root.title("Facial Recognition System")

# Create the needed variables
global image_names
global filesz
global encodeList
global attend_dict
global del_names,del_ind

path = "ImagesAttendance"
images = []  #  img to numpy array
encodeList=[]
filesz=tuple()
image_names = []  # Stores names
mylist = os.listdir(path)  # lists all the images in dir
savedImg = []
attend_dict={}
del_names=[]
del_ind=[]


# Accessing images in the folder
def access():
    global images,image_names
    for cl in mylist:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        image_names.append(os.path.splitext(cl)[0]) #for the main name use [0] for the extension use [1]
    print(image_names)
    image_names2 = image_names[:]





def clean():
    for f in os.listdir(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/only_name"):
        os.remove(fr"/home/b-r-arjun/Documents/Projects/Final Face Reco/only_name/{f}")




def find_encodings(images):
    for image in images:
    
    	#face_recognition library loads the picture in BGR form but to print it and for encodings we need RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        encode = face_recognition.face_encodings(image)[0] #It returns the encodings of the passed image.
        
        
        encodeList.append(encode)
    return encodeList




# To save the captured image
def save_img(imagesz,nami):
    savedImg=os.listdir(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/only_name")
    if nami not in savedImg:
        cv2.imwrite(fr"/home/b-r-arjun/Documents/Projects/Final Face Reco/only_name/{nami}.jpg", imagesz)


# To mark the attendace into txt file for a new name
def markAttendance(name):
    print(name, "attended")

    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()  # reads every line in attendance list

        for line in myDataList:
            line = line.strip()
            entry = line.split(',')
            attend_dict[entry[0]] = entry[1:]

        if name not in attend_dict.keys():
            now = datetime.now()
            dtString = now.strftime("%I:%M %p")  # I - 12 hr format() , minute , pm or am
            attend_dict[name] = [dtString,""]  # writes time

        elif name in attend_dict.keys():
            now = datetime.now()
            dtString = now.strftime("%I:%M %p")  # I - 12 hr format() , minute , pm or am
            attend_dict[name][1]=dtString


def webcam_scan():
    
    cap = cv2.VideoCapture(0) # starts video capture through webcam

    while True:
        
        # img = numpy array  ,  succces= if loaded or not
    
        success,img = cap.read()
    
        # we resize to 1/4th for ease of calculation and faster read time
        
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Find all the faces in the current frame in the webcam
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        # displays a text below                         no                 coordi           font            scale        colour    size
        cv2.putText(img,f'Number of faces detected: {len(facesCurFrame)}',(100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),2)

        # 
        for encodeFace,FaceLoc in zip(encodesCurFrame,facesCurFrame): #Grabs one encoding and one face 
            
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.5) # lower is more strict
            #takes an array of encodings and one encoding as parameters.
            
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            
            
            # This will compare the face distance values from the images in the directory to the 
            #webcam and returns the smallest value meaning the face that matches the most
            matchIndex = np.argmin(faceDis) 

            if matches[matchIndex]:
                
                name = image_names[matchIndex].upper() # Capitalizes each word
             
                # FaceLoc = up right down left
                y1,x2,y2,x1=FaceLoc
             
                # Multiply by 4 cuz we scaled down by  1/4
               
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4

                # Draws rectangle img , loc , colour , size
                cv2.rectangle(img, (x1,y1),(x2,y2) ,(255, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2),(255, 255, 0), cv2.FILLED)
                
                
                # Displays name
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                
                
                # Save the image if not saved already 
                save_img(img, name)

                # Calls attendace to add name
                markAttendance(name)

            
            else:
                    name = "UNKNOWN"
                    
                    # FaceLoc = up right down left
                    y1, x2, y2, x1 = FaceLoc
                    
                    # multiply by 4 cuz we scaled down to 1/4
                    
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    # draw's rectangle img , loc , colour , size
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
                    
                    # displays name
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        
        cv2.imshow('webcam',img)
        cv2.waitKey(1) # Continously display the image until key press.

        if keyboard.is_pressed('q'):
            print("i quit!!")
            cv2.destroyWindow('webcam')
            break



# Mark attendance
def attendance():
    ff = open("Attendance.csv", 'w+')
    ss = ""
    try:
        ff.writelines("NAME,ENTRY,EXIT,TIME_SPENT")
        ff.writelines("\n")
        del attend_dict['NAME']
        del attend_dict['UNKNOWN']
    except KeyError:
        print()

    for i in (attend_dict.keys()):
        ss += i
        entryy=attend_dict[i][0]
        exitt=attend_dict[i][1]
        try:
            ts=(int(exitt[3:-3]) - int(entryy[3:-3])) + (60*(int(exitt[:2]) - int(entryy[:2]) ))
            ss += "," + entryy + "," + exitt + "," + str(ts)
            ff.writelines(ss)
            ff.writelines("\n")
        except ValueError:
            print()

        ss = ""

    ff.close()
    os.startfile(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/Attendance.csv")






# Take a new picture from the webcam
def take_a_pic():
    new_name = pyautogui.prompt('What is your name?',title="Name",default="new_image")

    if new_name in del_names:
        loc=del_ind[del_names.index(new_name)]
        image_names[loc]=new_name

        new_name += ".jpg"
        tk.messagebox.showinfo("Alert", "Look at the Camera in 3 sec !")
        result, new_img = cv2.VideoCapture(0).read()
        cv2.imwrite(rf"/home/b-r-arjun/Documents/Projects/Final Face Reco/ImagesAttendance/{new_name}", new_img)
        cv2.imshow("New Image", new_img)
        cv2.waitKey(0)
        cv2.destroyWindow('New Image')

    else:
        new_name+= ".jpg"
        tk.messagebox.showinfo("Alert", "Look at the Camera in 3 sec !")
        result, new_img = cv2.VideoCapture(0).read()
        cv2.imwrite(rf"/home/b-r-arjun/Documents/Projects/Final Face Reco/ImagesAttendance/{new_name}",new_img)
        cv2.imshow("New Image",new_img)
        cv2.waitKey(0)
        cv2.destroyWindow('New Image')

        images.append(cv2.imread(fr'/home/b-r-arjun/Documents/Projects/Final Face Reco/ImagesAttendance/{new_name}'))
        image_names.append(os.path.splitext(new_name)[0])
        print(os.path.splitext(new_name)[0])
        encodeList.append(face_recognition.face_encodings(images[-1])[0])







def open_images_to_delete():
    L1 = image_names
    L2 = []
    li2 = os.listdir(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/ImagesAttendance")
    filesz = filedialog.askopenfilenames(title = "Select image files", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print("Selected files:", filesz)
    for xx in filesz:
        os.remove(xx)
        xx = os.path.splitext(xx[xx.find('nce') + 4:])[0]
        del_ind.append(L1.index(xx))
        del_names.append(image_names[L1.index(xx)])
        image_names[L1.index(xx)] = "unknown"
        print("removed : ", xx)

    set_dif = []
    for x in li2:
        L2.append(os.path.splitext(x)[0])
    set_dif = list(set(L1).symmetric_difference(set(L2)))
    set_dif = list(filter(lambda t: t != "unknown", set_dif))
    removed_names = ""
    for j in set_dif:
        removed_names += j + " , "
    tk.messagebox.showinfo("showinfo", f"Faces removed = {len(set_dif)}\n{removed_names}\nClose the Window")







def delete_a_face():
    root1 = tk.Toplevel()
    root1.geometry("310x220")
    root1.title("delete")
    image2 = PhotoImage(file='delete.png')
    bg1label = tk.Label(root1, image=image2, width=300, height=180)
    bg1label.pack()
    button9 = tk.Button(root1, text="Select the images", command=open_images_to_delete, width=300,pady=5)
    button9.pack()
    root1.mainloop()




def show():
    os.startfile(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/only_name")





def know_faces():
    os.startfile(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/ImagesAttendance")





def about():
    os.startfile(r"/home/b-r-arjun/Documents/Projects/Final Face Reco/ABOUT OURSELVES.png")







						#############----Mainn-------#############




clean() # Delete the images already scanned


access() # Get the names of images


encodeListKnown = find_encodings(images) # Encode all the images
print("Encoding Completed..")






# GUI with CustomTkinter


imag = tk.PhotoImage(file="bg4.png")

frame = cstk.CTkFrame(master=root)
frame.pack(padx=60,pady=20,fill="both",expand=True)

label = cstk.CTkLabel(master=frame,text="Facial Recognition System",font=("Roboto",24),compound="left")
label.pack(pady=12,padx=10)

bglabel = cstk.CTkLabel(master=frame,image=imag,text="", width=1080,height=1080)
bglabel.pack()

button1 = cstk.CTkButton(master=frame,text="Scan face (Webcam)",command=webcam_scan,height=80,width=250,font=("Arial",24))
button1.place(relx=0.3,rely=0.3,anchor="e")

button4 = cstk.CTkButton(master=frame,text="Known Images",command=know_faces,height=80,width=250,font=("Arial",24))
button4.place(relx=0.75,rely=0.3,anchor="w")

button5 = cstk.CTkButton(master=frame,text="Add a new face",command=take_a_pic,height=80,width=250,font=("Arial",24))
button5.place(relx=0.3,rely=0.57,anchor="e")

button6 = cstk.CTkButton(master=frame,text="Delete a face",command=delete_a_face,height=80,width=250,font=("Arial",24))
button6.place(relx=0.75,rely=0.562,anchor="w")

button3 = cstk.CTkButton(master=frame,text="About",command=about,height=80,width=250,font=("Arial",24))
button3.place(relx=0.3,rely=0.85,anchor="e")

button2 = cstk.CTkButton(master=frame,text="Show Scanned Images",command=show,height=80,width=250,font=("Arial",24))
button2.place(relx=0.75,rely=0.85,anchor="w")

button7 = cstk.CTkButton(master=frame,text="Open Attendance",command=attendance,height=80,width=250,font=("Arial",24))
button7.place(relx=0.52,rely=0.5,anchor="center")

root.mainloop()
