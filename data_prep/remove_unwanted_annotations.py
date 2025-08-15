import os

# Define base paths
src_ann_root = 'data/lemurattentiontarget/annotations'
dst_ann_root = 'data/lemurattentiontarget_small/annotations'
img_root = 'data/lemurattentiontarget_small/images'

# Walk through the train/test folders
for split in ['train', 'test']:
    split_dir = os.path.join(src_ann_root, split)

    for video_name in os.listdir(split_dir):
        video_path = os.path.join(split_dir, video_name)

        if not os.path.isdir(video_path):
            continue

        for cam_name in os.listdir(video_path):
            cam_path = os.path.join(video_path, cam_name)

            if not os.path.isdir(cam_path):
                continue

            for filename in os.listdir(cam_path):
                if not filename.endswith('.txt'):
                    continue

                src_txt_path = os.path.join(cam_path, filename)

                # Prepare destination path
                rel_path = os.path.join(split, video_name, cam_name, filename)
                dst_txt_path = os.path.join(dst_ann_root, rel_path)
                

                filtered_lines = []

                with open(src_txt_path, 'r') as infile:
                    for line in infile:
                        line = line.strip()
                        if not line:
                            continue

                        frame_name = line.split(',')[0]

                        # Build expected image path to check
                        expected_img_path = os.path.join(img_root, video_name, cam_name, frame_name)

                        if os.path.exists(expected_img_path):
                            filtered_lines.append(line)

                # Write filtered lines to destination
                if filtered_lines:
                    os.makedirs(os.path.dirname(dst_txt_path), exist_ok=True)
                    with open(dst_txt_path, 'w') as outfile:
                        outfile.write('\n'.join(filtered_lines) + '\n')
