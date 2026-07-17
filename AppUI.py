import json
import pickle
import threading
from collections import Counter, deque
from pathlib import Path
from queue import Empty, Queue

import cv2
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import messagebox


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Translator")
        self.root.geometry("700x560")
        self.root.configure(bg="#f4f7fb")
        self.root.resizable(False, False)

        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "models" / "alphabet_landmark_model.pkl"
        self.labels_path = self.base_dir / "models" / "alphabet_landmark_labels.json"

        self.model = None
        self.class_names = []
        self.cap = None
        self.hands = None
        self.running = False
        self.worker = None
        self.result_queue = Queue()

        self.prediction_buffer = deque(maxlen=5)

        self.feature_names = []
        for hand_num in range(2):
            for point_num in range(21):
                self.feature_names.extend([
                    f"h{hand_num+1}_x{point_num}",
                    f"h{hand_num+1}_y{point_num}",
                    f"h{hand_num+1}_z{point_num}",
                ])

        self.mode_var = tk.StringVar(value="ASL")
        self.prediction_var = tk.StringVar(value="No prediction")
        self.confidence_var = tk.StringVar(value="0.00")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._load_model()
        self._poll_results()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Real-Time Sign Language Translator",
            font=("Arial", 22, "bold"),
            bg="#f4f7fb",
            fg="#14213d"
        )
        title.pack(pady=20)

        mode_frame = tk.LabelFrame(
            self.root,
            text="Select Mode",
            font=("Arial", 12, "bold"),
            bg="#f4f7fb",
            fg="#111827",
            padx=25,
            pady=15
        )
        mode_frame.pack(pady=10)

        tk.Radiobutton(
            mode_frame,
            text="ASL",
            variable=self.mode_var,
            value="ASL",
            font=("Arial", 12),
            bg="#f4f7fb"
        ).pack(side="left", padx=25)

        tk.Radiobutton(
            mode_frame,
            text="ISL",
            variable=self.mode_var,
            value="ISL",
            font=("Arial", 12),
            bg="#f4f7fb"
        ).pack(side="left", padx=25)

        info_frame = tk.Frame(self.root, bg="#f4f7fb")
        info_frame.pack(pady=20)

        labels = [
            ("Prediction:", self.prediction_var, "#059669"),
            ("Confidence:", self.confidence_var, "#d97706"),
            ("Status:", self.status_var, "#2563eb"),
        ]

        for i, (label_text, var, color) in enumerate(labels):
            tk.Label(
                info_frame,
                text=label_text,
                font=("Arial", 16, "bold"),
                bg="#f4f7fb",
                fg="#111827"
            ).grid(row=i, column=0, sticky="w", padx=10, pady=12)

            tk.Label(
                info_frame,
                textvariable=var,
                font=("Arial", 16),
                bg="#f4f7fb",
                fg=color
            ).grid(row=i, column=1, sticky="w", padx=10, pady=12)

        note = tk.Label(
    self.root,
            text="Click Start Recognition to begin.",
            font=("Arial", 10),
            bg="#f4f7fb",
            fg="#4b5563"
        )
        note.pack(pady=5)


        button_frame = tk.Frame(self.root, bg="#f4f7fb")
        button_frame.pack(pady=20)

        self.start_btn = tk.Button(
            button_frame,
            text="Start Recognition",
            font=("Arial", 12, "bold"),
            bg="#10b981",
            fg="white",
            width=16,
            command=self.start_recognition
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(
            button_frame,
            text="Stop Recognition",
            font=("Arial", 12, "bold"),
            bg="#ef4444",
            fg="white",
            width=16,
            command=self.stop_recognition,
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            font=("Arial", 12, "bold"),
            bg="#374151",
            fg="white",
            width=16,
            command=self.close_app
        )
        exit_btn.grid(row=0, column=2, padx=10)

    def _load_model(self):
        if not self.model_path.exists() or not self.labels_path.exists():
            messagebox.showerror("Missing Files", "Model or label files not found.")
            self.root.destroy()
            return

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    def start_recognition(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam.")
            return

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

        self.running = True
        self.prediction_buffer.clear()
        self.status_var.set(f"{self.mode_var.get()} mode running")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.worker = threading.Thread(target=self._recognition_loop, daemon=True)
        self.worker.start()

    def stop_recognition(self):
        if not self.running:
            return

        self.running = False
        self.status_var.set("Stopping...")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.hands is not None:
            self.hands.close()
            self.hands = None

        cv2.destroyAllWindows()

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")

    def _recognition_loop(self):
        mp_draw = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.result_queue.put(("No frame", 0.0, "Camera read failed"))
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            display_label = "No hand detected"
            confidence = 0.0
            status = f"{self.mode_var.get()} mode running"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks_row = []
                for lm in hand_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])

                while len(landmarks_row) < 126:
                    landmarks_row.extend([0.0, 0.0, 0.0])

                input_df = pd.DataFrame([landmarks_row], columns=self.feature_names)

                probs = self.model.predict_proba(input_df)[0]
                pred_index = int(probs.argmax())
                confidence = float(probs.max())
                predicted_label = self.class_names[pred_index]

                self.prediction_buffer.append(predicted_label)
                most_common_label, count = Counter(self.prediction_buffer).most_common(1)[0]

                if confidence >= 0.35 or count >= 3:
                    display_label = most_common_label
                else:
                    display_label = "Uncertain"

            else:
                self.prediction_buffer.clear()

            self.result_queue.put((display_label, confidence, status))

            cv2.putText(frame, f"Mode: {self.mode_var.get()}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Prediction: {display_label}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Press Q to stop webcam", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Sign Language Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.result_queue.put(("No prediction", 0.0, "Stopped"))
        self.root.after(0, self.stop_recognition)

    def _poll_results(self):
        try:
            while True:
                label, confidence, status = self.result_queue.get_nowait()
                self.prediction_var.set(label)
                self.confidence_var.set(f"{confidence:.2f}")
                self.status_var.set(status)
        except Empty:
            pass

        self.root.after(100, self._poll_results)

    def close_app(self):
        self.stop_recognition()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
