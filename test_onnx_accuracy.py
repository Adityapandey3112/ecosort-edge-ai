import os
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((96, 96))
    
    img_data = np.array(img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def evaluate_onnx_model(model_path, val_dir):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_correct = 0
    total_samples = 0

    print(f"\n--- Running Independent ONNX Inference on '{val_dir}' ---")

    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(val_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder '{class_folder}' not found. Skipping.")
            continue

        class_correct = 0
        class_samples = 0

        for img_name in os.listdir(class_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_folder, img_name)
                input_tensor = preprocess_image(img_path)

                raw_outputs = session.run([output_name], {input_name: input_tensor})[0]
                predicted_class = int(np.argmax(raw_outputs, axis=1)[0])

                if predicted_class == class_idx:
                    class_correct += 1
                class_samples += 1

        if class_samples > 0:
            class_acc = 100 * class_correct / class_samples
            print(f"Class '{class_name}': {class_acc:.2f}% accuracy ({class_correct}/{class_samples})")

        total_correct += class_correct
        total_samples += class_samples

    if total_samples > 0:
        overall_acc = 100 * total_correct / total_samples
        print(f"\n✅ OVERALL ONNX VAL ACCURACY: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    else:
        print("❌ No valid images found in validation path.")

if __name__ == "__main__":
    import sys
    default_val_path = "/content/ecosort_dataset_public/val" if os.path.exists("/content/ecosort_dataset_public/val") else "data/val"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.onnx")
    parser.add_argument("--val_dir", type=str, default=default_val_path)
    
    if len(sys.argv) <= 1 or "colab_kernel_launcher" in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    evaluate_onnx_model(args.model, args.val_dir)
  
