import os
import cv2

input_folder = 'Temp_Images'
output_folder = 'Original_Images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    input_image_path = os.path.join(input_folder, filename)
    image = cv2.imread(input_image_path)
    height, width, _ = image.shape
    size = min(height, width)
    y = (height - size) // 2
    x = (width - size) // 2
    cropped_image = image[y:y+size, x:x+size]
    resized_image = cv2.resize(cropped_image, (512, 512))
    output_image_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_image_path, resized_image)
