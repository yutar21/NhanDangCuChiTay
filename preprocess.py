import cv2
import os

def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Resize image to img_size
        img = cv2.resize(img, img_size)

        # Convert image to grayscale (if needed)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values (if needed)
        # img = img / 255.0  # Example of normalization

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)

input_dir = 'D:\\HocTap\\Python\\NhanDienCuChiTay\\data'
output_dir = 'D:\\HocTap\\Python\\NhanDienCuChiTay\\traindata'
preprocess_images(input_dir, output_dir)
