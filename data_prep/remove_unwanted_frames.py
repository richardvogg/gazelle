import os
import shutil

# Define paths
# This script assumes the images are stored in a specific structure
#source
#|-- images
#|   |-- video1
#|   |   |-- frame1.jpg
#|   |   |-- frame2.jpg
#|   |-- video2
#|--annotations
#|   |-- train
#|   |   |-- video1
#|   |   |   |-- ind1.txt
#|   |   |   |-- ind2.txt
#|   |   |-- video2
#|   |-- test
#|   |   |-- video3
#|   |   |   |-- ind1.txt
#|   |   |   |-- ind2.txt
#...

#target
#|-- all_files_gaze.txt

source_root = 'data/lemurattentiontarget_new/'
target_root = 'data/lemurattentiontarget_new_clean/'

src_img_root = os.path.join(source_root, 'images')
dst_img_root = os.path.join(target_root, 'images')
src_ann_root = os.path.join(source_root, 'annotations')
dst_ann_root = os.path.join(target_root, 'annotations')
txt_file_path = os.path.join(target_root, 'all_files_gaze.txt')

# Read all image paths from txt file
with open(txt_file_path, 'r') as f:
    relative_paths = [line.strip() for line in f if line.strip()]

'''
# Process each file
for rel_path in relative_paths:
    # Remove leading "images/" if it exists
    rel_subpath = rel_path[len("images/"):] if rel_path.startswith("images/") else rel_path

    src = os.path.join(src_img_root, rel_subpath)
    dst = os.path.join(dst_img_root, rel_subpath)

    # Make sure destination folder exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Copy or move the file
    if os.path.exists(src):
        shutil.copy2(src, dst)  # Use shutil.move(src, dst) to move instead of copy
    else:
        print(f"Warning: Source file not found: {src}")

'''



# Walk through the train/test folders
for split in ['train', 'test']:
    split_dir = os.path.join(src_ann_root, split)

    for video_name in os.listdir(split_dir):
        video_path = os.path.join(split_dir, video_name)

        if not os.path.isdir(video_path):
            continue

        #for cam_name in os.listdir(video_path):
        #    cam_path = os.path.join(video_path, cam_name)
        #    print(cam_path)
        #    if not os.path.isdir(cam_path):
        #        continue

        for filename in os.listdir(video_path):
            
            if not filename.endswith('.txt'):
                continue

            src_txt_path = os.path.join(video_path, filename)

            # Prepare destination path
            rel_path = os.path.join(split, video_name, filename)
            dst_txt_path = os.path.join(dst_ann_root, rel_path)


            filtered_lines = []

            with open(src_txt_path, 'r') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    frame_name = line.split(',')[0]

                    # Build expected image path to check
                    expected_img_path = os.path.join(dst_img_root, video_name, frame_name)

                    if os.path.exists(expected_img_path):
                        filtered_lines.append(line)

            # Write filtered lines to destination
            if filtered_lines:
                os.makedirs(os.path.dirname(dst_txt_path), exist_ok=True)
                with open(dst_txt_path, 'w') as outfile:
                    outfile.write('\n'.join(filtered_lines) + '\n')
