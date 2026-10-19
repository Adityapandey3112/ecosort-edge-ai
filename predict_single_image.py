import argparse
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# Your dataset classes in exact alphabetical order
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def preprocess_image(image_path):
    """
    Preprocesses a raw input image to match the exact tensor format [1, 3, 96, 96]
    expected by the trained EcoSort ONNX model.
    """
    # 1. Open image and ensure 3-channel RGB format
    img = Image.open(image_path).convert("RGB")

    # 2. Resize spatially to 96x96 pixels
    img = img.resize((96, 96))

    # 3. Scale pixel values from [0, 255] to [0.0, 1.0]
    img_data = np.array(img, dtype=np.float32) / 255.0

    # 4. Standard ImageNet Mean & Standard Deviation Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    # 5. Transpose layout from HWC [96, 96, 3] to CHW [3, 96, 96]
    img_data = img_data.transpose(2, 0, 1)

    # 6. Add batch dimension: [1, 3, 96, 96]
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def softmax(x):
    """Computes probability distribution over logits."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def predict(image_path, model_path="model.onnx"):
    if not os.path.exists(model_path):
        print(f"❌ Error: ONNX model file '{model_path}' not found!")
        return

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return

    # Load ONNX Runtime session
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Preprocess image
    input_tensor = preprocess_image(image_path)

    # Execute model inference
    raw_logits = session.run([output_name], {input_name: input_tensor})[0]

    # Convert logits to probabilities
    probabilities = softmax(raw_logits)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASSES[predicted_idx]
    confidence = probabilities[predicted_idx] * 100

    # Display results
    print("\n--- EcoSort Single Image Prediction ---")
    print(f"Input Image : {image_path}")
    print(f"Prediction  : {predicted_class.upper()}")
    print(f"Confidence  : {confidence:.2f}%\n")

    print("Class Probabilities Breakdown:")
    for cls, prob in zip(CLASSES, probabilities):
        bar = "█" * int(prob * 20)
        print(f"  {cls:<10}: {prob * 100:6.2f}% | {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a trash image using model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Path to your photo (e.g., my_bottle.jpg)")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to .onnx file")
    args = parser.parse_args()

    predict(args.image, args.model)
