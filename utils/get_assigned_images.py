import argparse
import os
import shutil
import glob

def copy_my_images(member_name, path):
    """
    Copy a subset of images for a given team member.

    Parameters:
    namemember_name(str): The name of the team member.
    path (str): The path to the directory containing the images.

    Returns:
    None

    Raises:
    ValueError: If the member name of the team is not found in the list of members.
    """
    
    path = os.path.join(path, '*jpg')

    dest_dir = member_name + '_images'

    members = ['sobhy', 'rashed', 'nada', 'yomna', 'reem', 'boda']
    
    images = sorted(glob.glob(path))
    assigned_images = images[members.index(member_name)*530: (members.index(member_name)+1)*530]

    # create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # copy the images to the destination directory
    for image in assigned_images:
        shutil.copy(image, os.path.join(dest_dir, os.path.basename(image)))

if __name__ == '__main__':
    # setup argument parser
    parser = argparse.ArgumentParser(description='Copy images for a given team.')
    parser.add_argument('member_name', help='member_name of the team')
    parser.add_argument('path', help='path to the images')

    # parse arguments
    args = parser.parse_args()

    # call copy_my_images functionwith the parsed arguments
    copy_my_images(args.member_name, args.path)
