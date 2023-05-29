import os
import shutil
import glob
import argparse


def filter_object(object, images_path, annotations_path):
    
    # Initialize paths
    object_path = object
    if not images_path:
        images_path = './dataset/images'
    if not annotations_path:
        annotations_path = './dataset/annotations'

    # Get classes names 
    with open('./classes.txt', 'r') as file:
        classes = [class_name.strip()for class_name in file.readlines()]

    # Get id that represent filtered class 
    object_id = classes.index(object)

    # Create new directory with the name of the filtered class
    if os.path.exists(object):
        shutil.rmtree(object)

    os.makedirs(object)

    # Get annotations files
    annotations_files = sorted(glob.glob(os.path.join(annotations_path, '*txt')))

    """
    Iterate over every annotation file 
        and append annotation of filtered object to a new file with the same name
    """
    for annotation_file in annotations_files:
        file_name = annotation_file.split('/')[-1]
        image_name = file_name.replace('txt', 'jpg')
        new_file_path = os.path.join(object_path, file_name)

        with open(annotation_file, 'r') as file:
            for line in file:
                if line.startswith(str(object_id)):
                    with open(new_file_path, 'a') as new_file:
                        new_file.write(' '.join(['0'] + line.split()[1:]))
                        new_file.write('\n')

                if os.path.exists(new_file_path):
                    src = os.path.join(images_path, image_name)
                    dest = os.path.join(object_path, image_name)
                    shutil.copy(src, dest)

            # Remove last empty line
            if os.path.exists(new_file_path):

                with open(new_file_path, 'r') as new_file:
                    lines = new_file.readlines()

                with open(new_file_path, 'w') as new_file:
                    lines[-1] = lines[-1].strip()
                    new_file.writelines(lines)     

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Filter classes.')
    parser.add_argument('object', help='Object to be filtered')
    parser.add_argument('--images_path', default=None, help='Path of the images to be copied')
    parser.add_argument('--annotations_path', default=None, help='Path to the annotations')

    # Parse arguments
    args = parser.parse_args()

    # Filter objects
    filter_object(args.object, args.images_path, args.annotations_path)