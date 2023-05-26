import os
import shutil
import argparse

def move_annotated_images(images_path: str, annotations_path: str) -> None:
    """
    Moves annotated images from a given directory to images directory under dataset directory.

    Parameters:
    images_path (str): Path of the directory containing the images to be moved.
    annotations_path (str): Path of the directory containing the annotation files.

    Returns:
    None
    """
    os.makedirs('./dataset/images', exist_ok=True)

    dest_dir = './dataset/images'

    # Create a list of image filenames to move by replacing the '.txt' extension with '.jpg'
    images_to_move = [file.replace('txt', 'jpg') for file in os.listdir(annotations_path)]

    # Iterate over the list of image filenames and move each image
    for image in images_to_move:
        shutil.move(os.path.join(images_path, image), dest_dir)

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Delete annotated images.')
    parser.add_argument('images_path', help='Path of the images to be deleted')
    parser.add_argument('annotations_path', help='Path to the annotations')

    # Parse arguments
    args = parser.parse_args()

    # Delete annotated images
    move_annotated_images(args.images_path, args.annotations_path)