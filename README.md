# Group Photo Face Fixer

This project is designed to automatically detect faces in a group photo, analyze if someone's eyes are closed (or facial expression looks unhappy) — and fix it using **AI-powered inpainting** with Stable Diffusion.

---

## Solution Overview

**Problem:**  
In group photos, it’s hard to make everyone look perfect — someone always blinks or has a tired face.

**Solution:**  
This pipeline automatically detects faces using **MediaPipe Face Landmarker**, analyzes eye states, generates an inpainting mask for closed eyes, and uses **Stable Diffusion Inpainting** to fix them.  

---

## Technologies Used

- `mediapipe` — for face detection and landmark extraction.
- `OpenCV` — for image processing.
- `diffusers` + `Stable Diffusion Inpainting` — for AI-powered image editing.
- `PIL` — for image file operations.
- Python 3.11

---

## Installation

1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/group-photo-face-fixer.git
cd group-photo-face-fixer
```


2️⃣ Install dependencies:
``` pip install -r requirements.txt```

If using Google Colab:
Make sure you have enabled GPU (Runtime -> Change runtime type -> GPU)


## Usage Example

1️⃣ Detect faces & prepare mask:
 run in local machine:
 
 ``` python check_eyes.py```
 
The output must be like this(it`s just an example):
Знайдено облич: 6
Обличчя 0: Left eye open: False, Right eye open: False
Обличчя 1: Left eye open: False, Right eye open: False
Обличчя 2: Left eye open: False, Right eye open: False
Обличчя 3: Left eye open: False, Right eye open: False
Обличчя 4: Left eye open: False, Right eye open: False
Обличчя 5: Left eye open: False, Right eye open: False
Зображення та маска збережені: init_image.png, mask_image.png

2️⃣ Fix the photo using Stable Diffusion:
I don`t have GPU so I used Google colab, if you are able to use your local machine for running run the next command:

```python generate.py```


## Notes
You can customize the prompt in the inpainting step to get different facial expressions.
The project was tested on group photos with up to 10 faces.
You can replace the model in the inpainting step with other fine-tuned face models like SG161222/Realistic_Vision_V5.1_inpainting for higher realism.
