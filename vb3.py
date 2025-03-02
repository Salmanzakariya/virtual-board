import mediapipe as mp
import cv2
import numpy as np
import time

ml = 150  # Margin left for the toolbar
max_x, max_y = 300 + ml, 50  
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0
current_color = (0, 255, 255) 

def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    elif x < 250 + ml:
        return "erase"
    elif x < 300 + ml:
        return "color_picker"
    else:
        return "none"

def index_raised(yi, y9):
    return (y9 - yi) > 40

# Initialize Mediapipe Hands
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Create the toolbar with tool names
tools = np.zeros((max_y, max_x - ml, 3), dtype="uint8")
cv2.putText(tools, "Line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(tools, "Rect", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(tools, "Draw", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(tools, "Circle", (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(tools, "Erase", (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(tools, "Color", (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Create the color palette (right side)
color_palette = np.zeros((300, 50, 3), dtype="uint8")
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255)   # Magenta
]
for i, color in enumerate(colors):
    color_palette[i * 50:(i + 1) * 50, :] = color


mask = np.ones((480, 640, 3), dtype="uint8") * 255  # White background

# Start video capture
cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

           
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Current tool:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            
            if curr_tool == "color_picker":
                if 590 < x < 640 and 50 < y < 350:  
                    current_color = colors[(y - 50) // 50] 

           
            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), current_color, thick)
                    prevx, prevy = x, y
                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "line":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), current_color, thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), current_color, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), current_color, thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), current_color, thick)
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), current_color, thick)
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), current_color, thick)
                        var_inits = False

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)


    cv2.rectangle(frm, (10, 400), (60, 450), current_color, -1)
    cv2.putText(frm, "Selected", (70, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frm, "Color", (70, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

   
    frm = cv2.addWeighted(frm, 0.7, mask, 0.3, 0)
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)
    frm[50:350, 590:640] = color_palette  # Right-side palette
    cv2.putText(frm, curr_tool, (270 + ml, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Virtual Painter", frm)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()