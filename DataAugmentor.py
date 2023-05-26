import os
import cv2


def augment_image(image):
    augmented_images = []

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated_image)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(rotated_image)
    mirrored_image = cv2.flip(image, 1)
    augmented_images.append(mirrored_image)

    contrast_factors = [0.6, 1.4]
    brightness_factors = [0.6, 1.0, 1.4]
    saturation_factors = [0.6, 1.4]

    for brightness_factor in brightness_factors:
        for contrast_factor in contrast_factors:
            for saturation_factor in saturation_factors:
                augmented_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
                augmented_image[:, :, 1] = augmented_image[:, :, 1] * saturation_factor
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_HSV2BGR)
                augmented_images.append(augmented_image)
    return augmented_images


input_directory = "Original_Images"
output_directory = "Augmented_Images"

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        original_image = cv2.imread(os.path.join(input_directory, filename))
        augmented_images = augment_image(original_image)
        basename = os.path.splitext(filename)[0]
        for i, augmented_image in enumerate(augmented_images):
            output_filename = f"{basename}_{i}.jpg"
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, augmented_image)
