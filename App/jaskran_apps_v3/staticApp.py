import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOv8App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")

        # Create and place the title label at the top
        self.title_label = tk.Label(root, text="Facial Emotion Detection", font=("Helvetica", 12, "bold"))
        self.title_label.pack(pady=10)

        # Create and place widgets
        self.choose_button = tk.Button(root, text="Choose Image", command=self.choose_image)
        self.choose_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Load and display image
            self.display_image(file_path)
            # Perform detection
            self.perform_detection(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((800, 800))
        self.img = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.img)

    def perform_detection(self, file_path):
        # Load the YOLOv8 model
        model = YOLO("jaskaranOwnV3best.pt")  # Change to 'yolov8' if needed

        # Load image
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(img_rgb)

        confidences = []
        class_ids = []
        

        for result in results:
            boxes = result.boxes.cpu().numpy()

            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        if all(len(x) != 0 for x in confidences) and all(len(x) != 0 for x in class_ids):
            confidences = [y.item() for x in confidences for y in x]
            class_ids = [y.item() for x in class_ids for y in x]

            # Find the index of the highest confidence
            if confidences:
                max_conf_index = confidences.index(max(confidences))
                print(f"max conf: {max(confidences)}")
                


        new_results2 = model(img_rgb, conf=(max(confidences)-0.01))
        #self.display_annotated_image(new_results2)

        # Extract the annotated frame
        annotated_img = new_results2[0].plot()

        # Convert to PIL Image and display
        annotated_pil_image = Image.fromarray(annotated_img)
        annotated_pil_image.thumbnail((800, 800))
        self.annotated_img = ImageTk.PhotoImage(annotated_pil_image)
        self.image_label.config(image=self.annotated_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.mainloop()
