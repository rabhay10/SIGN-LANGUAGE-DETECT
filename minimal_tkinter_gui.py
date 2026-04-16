"""
Minimal Clean Tkinter GUI for ASL Converter
Simple, elegant design with same functionality
"""
import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import subprocess
import sys
import os
import enchant

from sign_language_predict import predict_single

# Color Scheme (Minimal White Theme)
BG_COLOR = "#f5f5f5"
CARD_BG = "#ffffff"
TEXT_COLOR = "#2c3e50"
ACCENT = "#3498db"
SUCCESS = "#27ae60"
DANGER = "#e74c3c"
BORDER = "#e0e0e0"

# Load resources
MODEL_PATH = "cnn8grps_rad1_model.h5"
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

try:
    dictionary = enchant.Dict("en_US")
except:
    dictionary = None

try:
    with open("wordlist.txt", "r") as f:
        autocomplete_words = [w.strip() for w in f.readlines()]
except FileNotFoundError:
    autocomplete_words = []

OFFSET = 29

class MinimalASLConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Converter")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1400x750+100+50")
        
        # State
        self.cap = cv2.VideoCapture(0)
        self.sentence = " "
        self.current_char = "—"
        self.suggestions = [" "] * 4
        self.history = [" "] * 10
        self.count = -1
        self.prev_char = ""
        
        self.create_ui()
        self.update_frame()
    
    def create_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_COLOR, height=60)
        header.pack(fill=tk.X, padx=30, pady=(20, 10))
        
        title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        tk.Label(header, text="Sign Language Converter", font=title_font,
                bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        
        # Content
        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Left: Camera
        cam_card = self._card(content, "Camera")
        cam_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.camera_label = tk.Label(cam_card, bg="#000")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Middle: Skeleton
        skel_card = self._card(content, "Hand Skeleton")
        skel_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.skeleton_label = tk.Label(skel_card, bg="#fff")
        self.skeleton_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Right: Output
        out_card = self._card(content, "Output")
        out_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Gesture
        gest_frame = tk.Frame(out_card, bg=ACCENT, height=80)
        gest_frame.pack(fill=tk.X, padx=15, pady=15)
        gest_frame.pack_propagate(False)
        
        self.gesture_label = tk.Label(gest_frame, text=self.current_char,
                                     font=("Helvetica", 40, "bold"),
                                     bg=ACCENT, fg="#fff")
        self.gesture_label.pack(expand=True)
        
        # Sentence
        tk.Label(out_card, text="Sentence", font=("Helvetica", 10),
                bg=CARD_BG, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, padx=15)
        
        self.sentence_text = tk.Text(out_card, height=5, font=("Courier", 12),
                                    bg="#fafafa", relief=tk.FLAT, padx=10, pady=10)
        self.sentence_text.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        # Buttons
        btn_frame = tk.Frame(out_card, bg=CARD_BG)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Button(btn_frame, text="Speak", command=self.speak, bg=SUCCESS,
                 fg="#fff", font=("Helvetica", 10, "bold"), relief=tk.FLAT,
                 padx=20, pady=8, cursor="hand2").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        tk.Button(btn_frame, text="Clear", command=self.clear, bg=DANGER,
                 fg="#fff", font=("Helvetica", 10, "bold"), relief=tk.FLAT,
                 padx=20, pady=8, cursor="hand2").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        
        # Suggestions
        tk.Label(out_card, text="Suggestions", font=("Helvetica", 10),
                bg=CARD_BG, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, padx=15)
        
        sugg_frame = tk.Frame(out_card, bg=CARD_BG)
        sugg_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.sugg_btns = []
        for i in range(4):
            btn = tk.Button(sugg_frame, text="—", command=lambda idx=i: self.apply_sugg(idx),
                          bg="#ecf0f1", fg=TEXT_COLOR, font=("Helvetica", 9),
                          relief=tk.FLAT, padx=10, pady=6, cursor="hand2")
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            self.sugg_btns.append(btn)
    
    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD_BG, relief=tk.SOLID, borderwidth=1, highlightbackground=BORDER)
        tk.Label(frame, text=title, font=("Helvetica", 11, "bold"),
                bg=CARD_BG, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, padx=15, pady=10)
        return frame
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_copy = np.array(frame)
            hands_list, _ = hd.findHands(frame, draw=False, flipType=True)
            
            if hands_list:
                hand = hands_list[0]
                x, y, w, h = hand["bbox"]
                y1 = max(0, y - OFFSET)
                y2 = min(frame.shape[0], y + h + OFFSET)
                x1 = max(0, x - OFFSET)
                x2 = min(frame.shape[1], x + w + OFFSET)
                crop = frame_copy[y1:y2, x1:x2]
                
                if crop.size > 0:
                    handz_list, _ = hd2.findHands(crop, draw=False, flipType=True)
                    if handz_list:
                        pts = handz_list[0]["lmList"]
                        white = np.ones((400, 400, 3), np.uint8) * 255
                        os_x, os_y = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                        
                        for start, end in [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
                            for t in range(start, end):
                                cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
                                       (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        
                        cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                        
                        for i in range(21):
                            cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)
                        
                        skel_img = ImageTk.PhotoImage(Image.fromarray(white).resize((280, 280)))
                        self.skeleton_label.configure(image=skel_img)
                        self.skeleton_label.image = skel_img
                        
                        ch1 = predict_single(pts, white, model)
                        if ch1:
                            self.current_char = str(ch1)
                            self.gesture_label.config(text=self.current_char)
                            
                            if ch1 == "next" and self.prev_char != "next":
                                prev_stable = self.history[(self.count - 2) % 10]
                                if prev_stable != "next":
                                    if prev_stable == "Backspace":
                                        self.sentence = self.sentence[:-1] if self.sentence else " "
                                    elif prev_stable == "Speak":
                                        self.speak()
                                    else:
                                        self.sentence += str(prev_stable)
                            
                            self.prev_char = ch1
                            self.count += 1
                            self.history[self.count % 10] = ch1
                            
                            self.sentence_text.delete("1.0", tk.END)
                            self.sentence_text.insert("1.0", self.sentence.strip())
                            self.update_suggestions()
            
            cam_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((380, 280)))
            self.camera_label.configure(image=cam_img)
            self.camera_label.image = cam_img
        
        self.root.after(30, self.update_frame)
    
    def update_suggestions(self):
        if self.sentence.strip():
            idx = self.sentence.rfind(" ")
            word = self.sentence[idx + 1:]
            if word.strip():
                word_lower = word.lower()
                suggestions = [w for w in autocomplete_words if w.startswith(word_lower)]
                if len(suggestions) < 4 and dictionary:
                    try:
                        for w_sug in dictionary.suggest(word):
                            if w_sug.lower() not in [s.lower() for s in suggestions]:
                                suggestions.append(w_sug)
                    except:
                        pass
                self.suggestions = (suggestions + [" "] * 4)[:4]
        
        for i, btn in enumerate(self.sugg_btns):
            btn.config(text=self.suggestions[i].strip() or "—")
    
    def speak(self):
        if self.sentence.strip():
            subprocess.Popen([sys.executable, "speak_script.py", self.sentence.strip()])
    
    def clear(self):
        self.sentence = " "
        self.sentence_text.delete("1.0", tk.END)
    
    def apply_sugg(self, idx):
        word = self.suggestions[idx].strip()
        if word:
            space_idx = self.sentence.rfind(" ")
            self.sentence = (self.sentence[:space_idx] + " " + word.upper()).strip() + " "
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", self.sentence.strip())
    
    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MinimalASLConverter(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
