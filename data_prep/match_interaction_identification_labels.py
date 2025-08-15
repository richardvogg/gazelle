import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


video_path_root = "/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/"
label_path = "data/raw_annotations/"
output_path  = "data/lemurattentiontarget_new"
os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "images_check"), exist_ok=True)

#"A_e5_c7", "A_e7_c6", "A_e7_c7", "A_e8_c6", "A_e8_c7", "A_e9_c7"
experiment_names = ["A_e10_c6", "A_e10_c7", "A_e11_c6", "A_e11_c7", "A_e12_c6", "A_e12_c7", "A_e13_c6", "A_e13_c7"]

for experiment_name in experiment_names:

    if experiment_name.startswith("A"):
        group = "Alpha"
    elif experiment_name.startswith("B"):
        group = "B"
    video_path = os.path.join(video_path_root, group, experiment_name +".mp4")


    df = pd.read_csv(f"{label_path}/{experiment_name}_behaviors.csv")


    df_int = df.assign(frame=df.apply(lambda row: np.arange(row["StartFrame"], row["EndFrame"] + 1), axis=1)).explode("frame").drop(columns=["StartFrame", "EndFrame"]).reindex(columns=["frame", "Subject", "Action", "Target"]).sort_values("frame")
    df_int["Action"] = df_int["Action"].str.strip()
    df_int = df_int.sort_values("frame").reset_index(drop=True)

    interaction_list = ["successful_lift", "unsuccessful_lift", "successful_slide", "unsuccessful_slide", "successful_push", "unsuccessful_push", "looking_at", "scrounging", "manipulating", "licking"]

    # Read identification
    file_path = f"{label_path}/{experiment_name}_identification.txt"

    with open(file_path, "r") as file:
        file_content = file.readlines()

    metadata = [line.strip("#").strip() for line in file_content if line.startswith("#")]
    metadata_dict = dict(item.split(": ") for item in metadata)

    username = metadata_dict["username"]
    editDate = metadata_dict["editDate"]
    orderedNames = metadata_dict["orderedNames"].split(", ")
    dataColumns = metadata_dict["dataColumns"].split(", ")

    df_id = pd.read_csv(file_path, skiprows=len(metadata), names=dataColumns).sort_values(["species", "trackId", "trackNumber"])
    df_id = df_id[df_id["nameOrder"].notna()]
    df_id["name"] = df_id["nameOrder"].astype(int).apply(lambda idx: orderedNames[idx])

    # Group df_int by frame to process all interactions in the same frame together
    for frame, group in df_int.groupby("frame"):
        subjects = group["Subject"].tolist()
        targets = group["Target"].tolist()
        names_needed = set(subjects + targets)

        id_rows = df_id[df_id["trackNumber"] == frame]
        names_found = set(id_rows["name"].values)

        # Check if all needed names are present in id_rows
        if names_needed.issubset(names_found):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            if not ret or img is None:
                print(f"Failed to read frame {frame}")
                cap.release()
                continue

            img_path = os.path.join(output_path, "images", f"{experiment_name}_frame{frame}.jpg")
            cv2.imwrite(img_path, img)

            yellow = (0, 255, 255)

            # Draw all subject and target boxes and arrows
            for _, row in group.iterrows():
                subject = row["Subject"]
                target = row["Target"]

                subject_box = id_rows[id_rows["name"] == subject].iloc[0, 2:6].values.astype(int)
                target_box = id_rows[id_rows["name"] == target].iloc[0, 2:6].values.astype(int)

                def draw_box(img, box, color, thickness=2):
                    x, y, w, h = box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

                def box_center(box):
                    x, y, w, h = box
                    return (int(x + w / 2), int(y + h / 2))

                draw_box(img, subject_box, yellow)
                draw_box(img, target_box, yellow)

                subject_center = box_center(subject_box)
                target_center = box_center(target_box)
                cv2.arrowedLine(img, subject_center, target_center, yellow, 2, tipLength=0.2)

            # Save image with overlays for checking
            img_check_path = os.path.join(output_path, "images_check", f"{experiment_name}_frame{frame}.jpg")
            print(img_check_path)
            cv2.imwrite(img_check_path, img)
            cap.release()
        else:
            missing = names_needed - names_found
            print(f"Frame {frame}: Missing {', '.join(missing)}")
