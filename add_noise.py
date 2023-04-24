from PIL import Image
import numpy as np
import os
import shutil
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


def add_gaussian_noise(image_path, mean=0, std=10):
    """
    Adds Gaussian noise to an image and returns the noisy image as a Numpy array.
    
    Args:
        image_path: The path to the input image.
        mean: The mean of the Gaussian distribution.
        std: The standard deviation of the Gaussian distribution.
    
    Returns:
        A Numpy array representing the noisy image.
    """
    # Open the image using Pillow.
    image = Image.open(image_path)

    # Convert the image to a Numpy array.
    image_array = np.array(image)

    # Generate a random matrix of the same shape as the image.
    noise = np.random.normal(mean, std, size=image_array.shape)

    # Add the noise to the image array.
    noisy_image_array = image_array + noise

    # Clip the values to the range [0, 255].
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert the Numpy array back to a Pillow image.
    noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))

    return noisy_image

def ghosting(input_path, num_ghosts=2, alpha=0.5, shift=10):
    # Create a copy of the input image
    # Create a copy of the input image
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # ghosted_img = np.zeros_like(img)

    # # Generate a random horizontal shift
    # dx = np.random.randint(-shift, shift)

    # # Apply the ghosting effect
    # if dx < 0:
    #     img_width = img.shape[1]
    #     img_shifted = np.hstack([img[:, -dx:], img[:, :img_width+dx]])
    #     ghosted_img[:, :img_width+dx] = alpha * img_shifted[:, -dx:] + (1 - alpha) * ghosted_img[:, :img_width+dx]
    # else:
    #     ghosted_img[:, dx:] = alpha * img[:, :-dx] + (1 - alpha) * ghosted_img[:, dx:]
    
    image = cv2.imread(input_path)

    # Create a copy of the original image to blend the ghosted images with
    augmented_image = image.copy()

    # Loop over the number of ghosted images to add
    for i in range(num_ghosts):
        # Shift the image by a random amount
        dx, dy = np.random.randint(-shift, shift, size=2)
        shifted_image = np.roll(image, dx, axis=1)
        shifted_image = np.roll(shifted_image, dy, axis=0)

        # Blend the shifted image with the original image using alpha blending
        blended_image = cv2.addWeighted(augmented_image, alpha, shifted_image, 1-alpha, 0)

        # Update the augmented image with the blended image
        ghosted = blended_image
        
    return Image.fromarray(ghosted)


# Set the input and output directories.
input_dir = "/home/mila/a/arkil.patel/scratch/rohan/GBDL/users-2/jianfeng/AtrialSeg-slice"
output_dir = "/home/mila/a/arkil.patel/scratch/rohan/GBDL/users-2/jianfeng/AtrialSeg-slice-ghost-10"
os.makedirs(output_dir, exist_ok=True)

# Set the mean and standard deviation of the Gaussian distribution.
mean = 0
std = 30

# Loop over all PNG files in the input directory.
for file_name in os.listdir(input_dir):
    for sub_file_name in os.listdir(os.path.join(input_dir, file_name)):
        # if sub_file_name contains the word mri then loop through the files in that folder
        if "mri" in sub_file_name:
            print(sub_file_name)
            for sub_sub_file_name in os.listdir(os.path.join(input_dir, file_name, sub_file_name)):

                # Construct the paths to the input and output files.
                input_path = os.path.join(input_dir, file_name, sub_file_name, sub_sub_file_name)
                # create the output path
                os.makedirs(os.path.join(output_dir, file_name, sub_file_name), exist_ok=True)
                output_path = os.path.join(output_dir, file_name, sub_file_name, sub_sub_file_name)
                print(input_path)
                print(output_path)
                # Add Gaussian noise to the image and save the result.
                # noisy_image = add_gaussian_noise(input_path, mean, std)
                # noisy_image.save(output_path)

                # ghosting atrifact

                ghost_image = ghosting(input_path)
                ghost_image.save(output_path)

        else:
            print(sub_file_name)
            # copy the folder to the output folder
            # os.makedirs(os.path.join(output_dir, file_name, sub_file_name), exist_ok=True)
            shutil.copytree(os.path.join(input_dir, file_name, sub_file_name), os.path.join(output_dir, file_name, sub_file_name))