import pickle as pkl
import numpy as np
import mediapipe as mp
import cv2

model_dict = pkl.load(open("model.pkl","rb"))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'A', 1:"B", 2:"C"}
while True:

    data_aux = []
    x_=[]
    y_=[]


    ret , frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style()
                                      )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y


                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_)*W)-10
        y1 = int(min(y_)*H)-10

        x2 = int(max(x_)*W)-10
        y2 = int(max(y_)*H)-10

        if(len(data_aux)!=42):
            continue
        prediction = model.predict([np.asarray(data_aux)])

        predictedCharacter = labels_dict[int(prediction[0])]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame,predictedCharacter,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,cv2.LINE_AA)


    cv2.imshow("Recognition",frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()