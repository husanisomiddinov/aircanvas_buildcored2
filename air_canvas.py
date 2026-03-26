import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class AirCanvas:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
            max_num_hands=1,
        )
        self.drawer = mp.solutions.drawing_utils

        self.colors = {
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255),
        }
        self.color_order = list(self.colors.keys())
        self.current_color_name = "Blue"
        self.current_color = self.colors[self.current_color_name]

        self.pinching = False
        self.pinch_threshold = 38

        self.current_stroke = []
        self.strokes = []
        self.points_buffer = deque(maxlen=6)

        self.ui_height = 80
        self.brush_thickness = 5

    @staticmethod
    def _to_pixel(landmark, frame_width, frame_height):
        return int(landmark.x * frame_width), int(landmark.y * frame_height)

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _smooth_point(self, point):
        self.points_buffer.append(point)
        mean_x = int(np.mean([p[0] for p in self.points_buffer]))
        mean_y = int(np.mean([p[1] for p in self.points_buffer]))
        return mean_x, mean_y

    def _draw_strokes(self, frame):
        for stroke in self.strokes:
            points = stroke["points"]
            color = stroke["color"]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, self.brush_thickness)

        if len(self.current_stroke) > 1:
            for i in range(1, len(self.current_stroke)):
                cv2.line(
                    frame,
                    self.current_stroke[i - 1],
                    self.current_stroke[i],
                    self.current_color,
                    self.brush_thickness,
                )

    def _draw_ui(self, frame):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, self.ui_height), (45, 45, 45), -1)

        x = 20
        for name in self.color_order:
            color = self.colors[name]
            selected = name == self.current_color_name
            radius = 20 if selected else 16
            thickness = -1 if selected else 2
            cv2.circle(frame, (x, 40), radius, color, thickness)
            cv2.putText(
                frame,
                name,
                (x - 24, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x += 85

        cv2.putText(
            frame,
            "Press C: clear   Press Q: quit",
            (w - 330, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Pinch threshold: {self.pinch_threshold}px",
            (w - 330, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _pick_color(self, fingertip):
        x, y = fingertip
        if y > self.ui_height:
            return

        start_x = 20
        for name in self.color_order:
            center = (start_x, 40)
            if self._distance((x, y), center) <= 24:
                self.current_color_name = name
                self.current_color = self.colors[name]
                self.pinching = False
                self.current_stroke = []
                self.points_buffer.clear()
                return
            start_x += 85

    def _finalize_stroke(self):
        if len(self.current_stroke) > 1:
            self.strokes.append(
                {"color": self.current_color, "points": self.current_stroke.copy()}
            )
        self.current_stroke = []
        self.points_buffer.clear()

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            pinch_distance = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                self.drawer.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape
                index_tip = self._to_pixel(
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    w,
                    h,
                )
                thumb_tip = self._to_pixel(
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
                    w,
                    h,
                )

                smooth_index_tip = self._smooth_point(index_tip)
                pinch_distance = self._distance(index_tip, thumb_tip)

                cv2.circle(frame, smooth_index_tip, 8, (255, 255, 255), -1)
                cv2.circle(frame, thumb_tip, 8, (200, 200, 200), -1)
                cv2.line(frame, smooth_index_tip, thumb_tip, (255, 255, 255), 2)

                self._pick_color(smooth_index_tip)

                if pinch_distance < self.pinch_threshold and smooth_index_tip[1] > self.ui_height:
                    if not self.pinching:
                        self.pinching = True
                        self.current_stroke = [smooth_index_tip]
                    else:
                        self.current_stroke.append(smooth_index_tip)
                else:
                    if self.pinching:
                        self._finalize_stroke()
                    self.pinching = False
            else:
                if self.pinching:
                    self._finalize_stroke()
                self.pinching = False

            self._draw_strokes(frame)
            self._draw_ui(frame)

            if pinch_distance is not None:
                cv2.putText(
                    frame,
                    f"Pinch distance: {int(pinch_distance)} px",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("AirCanvas", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                self.current_stroke = []
                self.strokes = []
                self.points_buffer.clear()
            if key == ord("1"):
                self.current_color_name = self.color_order[0]
                self.current_color = self.colors[self.current_color_name]
            if key == ord("2"):
                self.current_color_name = self.color_order[1]
                self.current_color = self.colors[self.current_color_name]
            if key == ord("3") and len(self.color_order) > 2:
                self.current_color_name = self.color_order[2]
                self.current_color = self.colors[self.current_color_name]
            if key == ord("4") and len(self.color_order) > 3:
                self.current_color_name = self.color_order[3]
                self.current_color = self.colors[self.current_color_name]

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AirCanvas().run()
