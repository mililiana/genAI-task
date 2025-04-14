import gradio as gr
import cv2
import numpy as np

def process_image(image, approach):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if approach == "Grayscale":
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  
    elif approach == "Edge Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif approach == "Blur":
        result = cv2.GaussianBlur(img, (15, 15), 0)
    else:
        result = img 

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

approaches = ["Grayscale", "Edge Detection", "Blur"]

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload a Photo"),
        gr.Radio(choices=approaches, label="Select Approach")
    ],
    outputs=gr.Image(type="numpy", label="Processed Photo"),
    title="Image Processing Demo",
    description="Upload a photo and choose a processing approach."
)

demo.launch()
