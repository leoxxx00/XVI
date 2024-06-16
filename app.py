import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import FineTunedResNet
import time

# Define the transform for the input image
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the trained ResNet50 model
model = FineTunedResNet(num_classes=3)
model.load_state_dict(torch.load('/content/lung_disease_detection/models/final_fine_tuned_resnet50.pth',
                                 map_location=torch.device('cpu')))
model.eval()


# Define a function to make predictions
def predict(image):
    start_time = time.time()  # Start the timer
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        top_prob, top_class = torch.topk(probabilities, 3)
        classes = ['🦠 COVID', '🫁 Normal', '🦠 Pneumonia']  # Adjust based on the classes in your model

    end_time = time.time()  # End the timer
    prediction_time = end_time - start_time  # Calculate the prediction time

    # Format the result string
    result = f"Top Predictions:\\n"
    for i in range(top_prob.size(0)):
        result += f"{classes[top_class[i]]}: {top_prob[i].item() * 100:.2f}%\\n"
    result += f"Prediction Time: {prediction_time:.2f} seconds"

    return result


# Example images with labels
examples = [
    ['examples/Pneumonia/02009view1_frontal.jpg', '🦠 Pneumonia'],
    ['examples/Pneumonia/02055view1_frontal.jpg', '🦠 Pneumonia'],
    ['examples/Pneumonia/03152view1_frontal.jpg', '🦠 Pneumonia'],
    ['examples/COVID/11547_2020_1200_Fig3_HTML-a.png', '🦠 COVID'],
    ['examples/COVID/11547_2020_1200_Fig3_HTML-b.png', '🦠 COVID'],
    ['examples/COVID/11547_2020_1203_Fig1_HTML-b.png', '🦠 COVID'],
    ['examples/Normal/06bc1cfe-23a0-43a4-a01b-dfa10314bbb0.jpg', '🫁 Normal'],
    ['examples/Normal/08ae6c0b-d044-4de2-a410-b3cf8dc65868.jpg', '🫁 Normal'],
    ['examples/Normal/IM-0178-0001.jpeg', '🫁 Normal']
]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray Image"),
    outputs=gr.Label(label="Predicted Disease"),
    examples=examples,
    title="Lung Disease Detection XVI",
    description="Upload a chest X-ray image to detect lung diseases such as 🦠 COVID-19, 🦠 Pneumonia, or 🫁 Normal. Use the example images to see how the model works."
)

# Launch the interface
interface.launch()