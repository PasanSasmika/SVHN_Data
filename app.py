# app.py

import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr
import numpy as np
from PIL import Image
import time

# --- IMPORTANT: The model class definition remains the same ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# --- Load Model and Define Transform (Same as before) ---
MODEL_PATH = "best_svhn_model.pth"
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- UPGRADED Prediction Function ---
def predict(image):
    """
    Takes a user-uploaded image and returns detailed prediction results.
    """
    # Start timer
    start_time = time.time()
    
    # Get original image dimensions for stats
    try:
        orig_h, orig_w, _ = image.shape
    except Exception:
        return None, None, "Error: Image could not be processed."

    # Preprocess: Convert to PIL, resize to 32x32
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB').resize((32, 32))
    
    # Transform for the model
    image_tensor = transform(pil_image).unsqueeze(0)

    # --- Model Prediction ---
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Stop timer and calculate duration
    duration = time.time() - start_time
    
    # --- Format Outputs for Gradio ---
    
    # 1. Create confidence scores for the BarPlot
    confidences = {str(i): prob.item() for i, prob in enumerate(probabilities)}
    
    # 2. Get the preprocessed image to display back to the user
    preprocessed_image_display = np.array(pil_image)
    
    # 3. Create a stats string
    stats_string = (
        f"Original Dimensions: {orig_w}x{orig_h}\n"
        f"Prediction Time: {duration:.3f} seconds"
    )
    
    return confidences, preprocessed_image_display, stats_string


# --- UPGRADED Gradio Interface using Blocks for more control ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as iface:
    gr.Markdown("# 🔢 Enhanced SVHN House Number Classifier")
    gr.Markdown("Upload an image or use the examples to see the model's prediction, confidence scores, and performance stats.")
    
    with gr.Row():
        # --- INPUT COLUMN ---
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload a House Number Image")
            submit_btn = gr.Button("Classify Digit", variant="primary")
            
            gr.Examples(
                examples=[["examples/1.png"], ["examples/3.png"], ["examples/8.png"]],
                inputs=input_image,
                fn=predict,
                label="Example Images"
            )

        # --- OUTPUT COLUMN ---
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Prediction Results")
            # Use gr.BarPlot for a richer visualization
            output_plot = gr.BarPlot(x="Digit", y="Confidence", title="Prediction Confidence", y_lim=[0, 1], color_map="blues_d")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🖼️ Preprocessed Image (32x32)")
                    # Show the image the model actually sees
                    output_proc_image = gr.Image(label="Model Input", height=128, width=128)
                with gr.Column():
                    gr.Markdown("### ⚙️ Performance Stats")
                    # Display the stats string
                    output_stats = gr.Textbox(label="Details", lines=2, interactive=False)

    # Connect the button to the function
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_plot, output_proc_image, output_stats]
    )

print("🚀 Launching Enhanced Gradio App...")
iface.launch()