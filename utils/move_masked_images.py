import os
import shutil
import argparse

def move_annotated_images(images_path: str, masks_path: str) -> None:
    """
    Moves annotated images from a given directory to images directory under dataset directory.

    Parameters:
    images_path (str): Path of the directory containing the images to be moved.
    annotations_path (str): Path of the directory containing the annotation files.

    Returns:
    None
    """
    os.makedirs('./dataset/images_for_segmantation', exist_ok=True)

    src_dir = images_path
    dest_dir = './dataset/images_for_segmantation'

    # Create a list of image filenames to move by replacing the extension with '.jpg'
    images_to_move = [file.split('.')[0] + '.jpg' for file in os.listdir(masks_path)]

    # Iterate over the list of image filenames and move each image
    for image in images_to_move:
        # Check if image exists in source directory
        if os.path.exists(os.path.join(src_dir, image)):
            # Move image if not exist in the destination directory
            if not os.path.exists(os.path.join(dest_dir, image)):
                shutil.move(os.path.join(src_dir, image), dest_dir)
            # Remove it from source directory if it already exists in destination directory
            else:
                os.remove(os.path.join(src_dir, image))


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Move annotated images.')
    parser.add_argument('images_path', help='Path of the images to be moved')
    parser.add_argument('masks_path', help='Path to the masked_images')

    # Parse arguments
    args = parser.parse_args()

    # Move annotated images
    move_annotated_images(args.images_path, args.masks_path)