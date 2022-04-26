import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import csv
import time
import os
import mediapipe as mp
import numpy as np
import dlib
import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import random
import keyboard
import pyautogui

newPath = 'Mcqs'
myList = os.listdir(newPath)
start = 0
end = len(myList)-1
mcq_no = random.randint(start, end)

Id = 0
name = ""

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
detector_face = dlib.get_frontal_face_detector()


def cal_mouth(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance


landmark_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

mouth_open = 30
ptime = 0

def test(Id, name):
    class MCQ():
        def __init__(self, data):
            self.question = data[0]
            self.choice1 = data[1]
            self.choice2 = data[2]
            self.choice3 = data[3]
            self.choice4 = data[4]
            self.answer = int(data[5])
            self.previous = data[6]
            self.userAns = None

        def update(self, cursor, bboxs):
            for x, bbox in enumerate(bboxs):
                x1, y1, x2, y2 = bbox
                if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                    self.userAns = x + 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)

    pathCsv = f'Mcqs/{myList[mcq_no]}'
    with open(pathCsv, newline='\n') as f:
        reader = csv.reader(f)
        dataAll = list(reader)[1:]

    mcqList = []
    for q in dataAll:
        mcqList.append(MCQ(q))

    print(len(mcqList))
    qNo = 0
    qTotal = len(dataAll)
    print(qTotal)

    startTime = time.time()
    timeTaken = 0
    typed = 0

    while True:

        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector_face(gray)

        # mouth movement detection and face count
        i = 0
        for face in faces:

            i = i + 1

            # print(face, i)
            if (i > 1):
                noOfFile = len(os.listdir("MultipleFaces")) + 1
                cv2.imwrite("MultipleFaces\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, 'face num' + str(i), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f'Only one person must be present!!', (150, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 2)

            # ----------Detect Landmarks-----------#
            shapes = landmark_model(gray, face)
            shape = face_utils.shape_to_np(shapes)

            # -------Detecting/Marking the lower and upper lip--------#
            lip = shape[48:60]
            # cv2.drawContours(img, [lip], -1, (0, 165, 255), thickness=3)

            # -------Calculating the lip distance-----#
            lip_dist = cal_mouth(shape)
            # print(lip_dist)

            if lip_dist > mouth_open:
                cv2.putText(img, f'Dont Talk!', (img.shape[1] // 2 - 170, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 2)
                noOfFile = len(os.listdir("Talking")) + 1
                cv2.imwrite("Talking\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)



        img.flags.writeable = False

        results = face_mesh.process(img)

        img.flags.writeable = True

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = img.shape
        face_3d = []
        face_2d = []

        image = pyautogui.screenshot()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        keyboard.on_press_key("a", lambda _: image.save(str(Id) + '-' + str(typed) + 'a'+'.jpg'))
        typed += 1
        keyboard.on_press_key("b", lambda _: image.save(str(Id) + '-' + str(typed) + 'b'+'.jpg'))
        typed += 1
        keyboard.on_press_key("c", lambda _: image.save(str(Id) + '-' + str(typed) + 'c' + '.jpg'))
        typed += 1
        keyboard.on_press_key("d", lambda _: image.save(str(Id) + '-' + str(typed) + 'd' + '.jpg'))
        typed += 1
        keyboard.on_press_key("e", lambda _: image.save(str(Id) + '-' + str(typed) + 'e' + '.jpg'))
        typed += 1
        keyboard.on_press_key("f", lambda _: image.save(str(Id) + '-' + str(typed) + 'f' + '.jpg'))
        typed += 1
        keyboard.on_press_key("g", lambda _: image.save(str(Id) + '-' + str(typed) + 'g' + '.jpg'))
        typed += 1
        keyboard.on_press_key("h", lambda _: image.save(str(Id) + '-' + str(typed) + 'h' + '.jpg'))
        typed += 1
        keyboard.on_press_key("i", lambda _: image.save(str(Id) + '-' + str(typed) + 'i' + '.jpg'))
        typed += 1
        keyboard.on_press_key("j", lambda _: image.save(str(Id) + '-' + str(typed) + 'j' + '.jpg'))
        typed += 1
        keyboard.on_press_key("k", lambda _: image.save(str(Id) + '-' + str(typed) + 'k' + '.jpg'))
        typed += 1
        keyboard.on_press_key("l", lambda _: image.save(str(Id) + '-' + str(typed) + 'l' + '.jpg'))
        typed += 1
        keyboard.on_press_key("m", lambda _: image.save(str(Id) + '-' + str(typed) + 'm' + '.jpg'))
        typed += 1
        keyboard.on_press_key("n", lambda _: image.save(str(Id) + '-' + str(typed) + 'n' + '.jpg'))
        typed += 1
        keyboard.on_press_key("o", lambda _: image.save(str(Id) + '-' + str(typed) + 'o' + '.jpg'))
        typed += 1
        keyboard.on_press_key("p", lambda _: image.save(str(Id) + '-' + str(typed) + 'p' + '.jpg'))
        typed += 1
        keyboard.on_press_key("r", lambda _: image.save(str(Id) + '-' + str(typed) + 'r' + '.jpg'))
        typed += 1
        keyboard.on_press_key("s", lambda _: image.save(str(Id) + '-' + str(typed) + 's' + '.jpg'))
        typed += 1
        keyboard.on_press_key("t", lambda _: image.save(str(Id) + '-' + str(typed) + 't' + '.jpg'))
        typed += 1
        keyboard.on_press_key("u", lambda _: image.save(str(Id) + '-' + str(typed) + 'u' + '.jpg'))
        typed += 1
        keyboard.on_press_key("v", lambda _: image.save(str(Id) + '-' + str(typed) + 'v' + '.jpg'))
        typed += 1
        keyboard.on_press_key("w", lambda _: image.save(str(Id) + '-' + str(typed) + 'w' + '.jpg'))
        typed += 1
        keyboard.on_press_key("x", lambda _: image.save(str(Id) + '-' + str(typed) + 'x' + '.jpg'))
        typed += 1
        keyboard.on_press_key("y", lambda _: image.save(str(Id) + '-' + str(typed) + 'y' + '.jpg'))
        typed += 1
        keyboard.on_press_key("z", lambda _: image.save(str(Id) + '-' + str(typed) + 'z' + '.jpg'))
        typed += 1



        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

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

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
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
                z = angles[2] * 360

                # See where the user's head tilting
                if face_3d.any():
                    if y < -3:
                        text = "Looking Left"
                        noOfFile = len(os.listdir("Left")) + 1
                        cv2.imwrite("Left\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)

                    elif y > 3:
                        text = "Looking Right"
                        noOfFile = len(os.listdir("Right")) + 1
                        cv2.imwrite("Right\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)

                    elif x < -3:
                        text = "Looking Down"
                        noOfFile = len(os.listdir("Down")) + 1
                        cv2.imwrite("Down\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)

                    elif x > 3:
                        text = "Looking Up"
                        noOfFile = len(os.listdir("Up")) + 1
                        cv2.imwrite("Up\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)

                    elif (x < 3 and x > -3 and y < 3 and y > -3):
                        text = "Forward"

                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        else:
            text = "No One"
            noOfFile = len(os.listdir("No-one")) + 1
            cv2.imwrite("No-one\Id" + str(Id) + "-" + str(noOfFile) + ".jpg", img)

            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        if qNo < qTotal:
            mcq = mcqList[qNo]

            # offset will provide thickness to the rectangle
            img, bbox = cvzone.putTextRect(img, mcq.question, [90, 145], 2, 2, offset=50, border=2)
            img, bbox1 = cvzone.putTextRect(img, mcq.choice1, [100, 275], 2, 2, offset=50, border=2)
            img, bbox2 = cvzone.putTextRect(img, mcq.choice2, [800, 275], 2, 2, offset=50, border=2)
            img, bbox3 = cvzone.putTextRect(img, mcq.choice3, [100, 530], 2, 2, offset=50, border=2)
            img, bbox4 = cvzone.putTextRect(img, mcq.choice4, [800, 530], 2, 2, offset=50, border=2)
            img, bbox5 = cvzone.putTextRect(img, mcq.previous, [1050, 50], 2, 2, offset=50, border=2)

            if hands:
                lmList = hands[0]['lmList']
                cursor = lmList[8]
                length, info = detector.findDistance(lmList[8], lmList[12])
                if length < 50:
                    mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4, bbox5])
                    if mcq.userAns is not None and mcq.userAns != 5:
                        time.sleep(0.8)
                        qNo += 1
                    if mcq.userAns == 5:
                        time.sleep(0.8)
                        qNo -= 1
        else:
            score = 0
            for mcq in mcqList:
                if mcq.answer == mcq.userAns:
                    score = score + 1
            totalscore = ((score / qTotal) * 100)
            img, _ = cvzone.putTextRect(img, f'Your Score: {score}/{qTotal}', [950, 400], 2, 2, offset=50,
                                        border=5)
            if cv2.waitKey(1) == ord('q'):
                break

        endTime = time.time()
        timeTaken = endTime - startTime
        img, _ = cvzone.putTextRect(img, f'Time: {int(timeTaken)}s', [1050, 635], 2, 2, offset=50, border=2)
        barValue = 25 + (950 // qTotal) * qNo
        cv2.rectangle(img, (25, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (25, 600), (975, 650), (255, 0, 255), 5)
        img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [500, 700], 2, 2, offset=16)
        # Id = (txt.get())
        # name = (txt2.get())

        cv2.imshow("Img", img)
        cv2.waitKey(1)
        end = time.time()

    row = [Id, name, totalscore, int(timeTaken)]
    with open('read.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


def main():

    st.set_page_config(page_title="Examination", page_icon="book", layout="wide")

    hide_menu_style = """
                    <style>
                    #MainMenu {visibility: hidden; }
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    selected = option_menu(None, ["Exam", "Instructions", 'Statistics'],
                            icons=["clock", "list-task", 'bar-chart'],
                            menu_icon="cast", default_index=0, orientation="horizontal",
                            styles={
                                "container": {"padding": "0!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"},
                                "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                                             "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "green"},
                            }
    )

    cola, colb, colc = st.columns([1, 4, 1])
    with cola:
        st.write("")
    with colb:
        st.sidebar.title('TABS')
    with colc:
        st.write("")
    st.sidebar.header("1: Exam")
    st.sidebar.write("This tab is for giving your exam")
    st.sidebar.header("2: Instructions")
    st.sidebar.write("This tab is for understanding what is necessary before you give your exam. First visit this tab")
    st.sidebar.header("3: Statistics")
    st.sidebar.write("This tab is for viewing statistics")

    html_temp = """
        <body>
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">OpenCv Based Virtual Quiz</h2>
        </div>
        </body>
        """

    st.markdown(html_temp, unsafe_allow_html=True)

    st.balloons()
    if(selected == "Exam"):
        st.title("Best of Luck!!!")

        col1, col2, col3 = st.columns([2, 4, 2])
        with col1:
            st.write("")
        with col2:
            Id = st.text_input("Enter your id")
        with col3:
            st.write("")

        col4, col5, col6 = st.columns([2, 4, 2])
        with col4:
            st.write("")
        with col5:
            name = st.text_input("Enter your Name")
        with col6:
            st.write("")

        col7, col8, col9 = st.columns([2, 4, 2])
        with col7:
            st.write("")
        with col8:
            if st.button("START THE EXAMINATION"):
                result = test(Id, name)
                st.subheader(f'{name} you have successfully submitted your test. Thank you!!!')
        with col9:
            st.write("")


        cap.release()
        cv2.destroyAllWindows()

    if(selected == "Instructions"):
        st.title("Instructions")
        st.subheader("1: First Enter you Name and press enter.")
        st.subheader("2: Repeat the same for Id.")
        st.subheader("3: Then click on start which will start the test.")
        st.subheader("4: While selecting the optins you simply have to use your index and middle finger and cut like a scissor which will select the desired option.")
        st.subheader("5: You will have four options along with a previous option button which is placed in order that if you want to change the selected option, you can.")
        st.subheader("6: The test is automatically proctored.")
        st.subheader("7: Here, you cannot leave the place till the test is completed. If no one is detected in front of the camera, you and the particular authority will be notified about the same. Your image will basically be saved.")
        st.subheader("8: Only you must be present before the camera. Someone other than you detected before the camera will be notified to the required authorities.")
        st.subheader("9: You must be facing the camera for the whole duration of the test. If you will be looking somewhere else other than the screen then it will be noted.")
        st.subheader("10: Also you must not talk while giving the test as it will be detected as well.")
        st.subheader("11: Do not try to search for the question's answer as your keyboard typing will be detected.")
        st.subheader("12: Screenshots of what you have been trying to search will be taken so do not type anything as there won't be any necessity for you to type anything forgiving the answer.")
        st.subheader("11: After your test is completed your score and time taken for the test will be shown.")
        st.subheader("12: If you want to check if your test has been submitted along with other student's submission details, you can visit the statistics tab where the results will be displayed. Since your's is the latest one look at the bottom most row for your details.")
        st.subheader("13: To check whether you have passed your test, check the color of the Score column. If it is green, then you have successfully passed your test.")

    if (selected == "Statistics"):
        st.subheader("1: About the Data")
        df = pd.read_csv('read.csv')

        def color_df(val):
            if val > 50:
                color = 'green'
            else:
                color = 'red'

            return f'background-color: {color}'

        st.dataframe(df.style.applymap(color_df, subset=[' Score']))

        st.subheader('2: Time wanted for each student')
        @st.cache
        def load_data(nrows):
            data = pd.read_csv('read.csv', nrows=nrows)
            return data

        wanted_data = load_data(1000)
        st.bar_chart(wanted_data[' Time'])

        df1 = pd.DataFrame(wanted_data[:], columns=[' Time', ' Score'])

        st.subheader("3: Relation between Time and Score")
        st.line_chart(df1)

        fig = plt.figure(figsize=(10, 4))
        plt.scatter(wanted_data[' Time'], wanted_data[' Score'])

        st.pyplot(fig)


if __name__ == '__main__':
    main()
