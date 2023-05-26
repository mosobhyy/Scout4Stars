import os
import argparse

def remove_annotated_images(images_path: str, annotations_path: str) -> None:
    """
    Deletes annotated images from a given directory.

    Parameters:
    images_path (str): Path of the directory containing the images to be deleted.
    annotations_path (str): Path of the directory containing the annotation files.

    Returns:
    None
    """
    # Create a list of image filenames to delete by replacing the '.txt' extension with '.jpg'
    images_to_delete = [file.replace('txt', 'jpg') for file in os.listdir(annotations_path)]

    # Iterate over the list of image filenames and delete each image
    for image in images_to_delete:
        os.remove(os.path.join(images_path, image))

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Delete annotated images.')
    parser.add_argument('images_path', help='Path of the images to be deleted')
    parser.add_argument('annotations_path', help='Path to the annotations')

    # Parse arguments
    args = parser.parse_args()

    # Delete annotated images
    remove_annotated_images(args.images_path, args.annotations_path)