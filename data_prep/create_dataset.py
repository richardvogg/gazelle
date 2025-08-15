import os
import pandas as pd
import numpy as np

image_path_root = "data/lemurattentiontarget_new/images/"
label_path = "data/raw_annotations/"
output_path = "data/lemurattentiontarget_new/annotations/"

# List all files in the image directory
image_files = []
for folder in os.listdir(image_path_root):
    folder_path = os.path.join(image_path_root, folder)
    if os.path.isdir(folder_path):
        image_files += [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
test_files = ['A_e5_c7', 'A_e10_c7', 'A_e11_c6']

# Extract unique base filenames (before '_frame')
base_names = set()
for fname in image_files:
    if '_frame' in fname:
        base = fname.split('_frame')[0]
        base_names.add(base)

# Create a folder for each unique base name in the output path
for base in base_names:

    if base in test_files:
        folder_path = os.path.join(output_path, "test", base)
    else:
        folder_path = os.path.join(output_path, "train", base)

    os.makedirs(folder_path, exist_ok=True)


    # Find all frame numbers for files that start with the current base
    frame_numbers = []
    for fname in image_files:
        if fname.startswith(base + "_frame"):
            try:
                frame_str = fname.split("_frame")[1].split(".")[0]
                frame_num = int(frame_str)
                frame_numbers.append(frame_num)
            except (IndexError, ValueError):
                continue
    frame_numbers = sorted(frame_numbers)

    # Interaction Labels
    df = pd.read_csv(f"{label_path}/{base}_behaviors.csv")

    df_int = df.assign(frame=df.apply(lambda row: np.arange(row["StartFrame"], row["EndFrame"] + 1), axis=1)).explode("frame").drop(columns=["StartFrame", "EndFrame"]).reindex(columns=["frame", "Subject", "Action", "Target"]).sort_values("frame")
    df_int["Action"] = df_int["Action"].str.strip()
    df_int = df_int.sort_values("frame").reset_index(drop=True)

    print(df_int.head())

    


    # Tracking and identification Labels
    file_path = f"{label_path}/{base}_identification.txt"
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

    df_id = df_id[df_id["trackNumber"].isin(frame_numbers)]
    for idx, name in enumerate(df_id["name"].unique()):
        fname = f"s{idx:02d}.txt"
        file_out_path = os.path.join(folder_path, fname)
        df_id_name = df_id[df_id["name"] == name]
        # Filter df_int for rows where this name is the subject
        df_int_subj = df_int[df_int["Subject"] == name]
        
        with open(file_out_path, "w") as f:
            for _, row in df_id_name.iterrows():
                frame = int(row["trackNumber"])
                image_file = f"{base}_frame{frame}.jpg"
                
                # Bounding box coordinates
                tlbr = [
                    int(row["xCoord"]),
                    int(row["yCoord"]),
                    int(row["xCoord"] + row["width"]),
                    int(row["yCoord"] + row["height"]),  # fixed 'right' to 'yCoord'
                ]

                # Default target_x and target_y if no interaction
                target_x = -1
                target_y = -1

                # Check if this frame has an interaction for this subject
                interaction_row = df_int_subj[df_int_subj["frame"] == frame]
                if not interaction_row.empty:
                    target_name = interaction_row.iloc[0]["Target"]

                    # Find target in df_id for the same frame
                    df_target = df_id[(df_id["name"] == target_name) & (df_id["trackNumber"] == frame)]
                    if not df_target.empty:
                        target_row = df_target.iloc[0]
                        target_x = int(target_row["xCoord"] + target_row["width"] / 2)
                        target_y = int(target_row["yCoord"] + target_row["height"] / 2)

                f.write(f"{image_file},{tlbr[0]},{tlbr[1]},{tlbr[2]},{tlbr[3]},{target_x},{target_y}\n")

            print(f"Processed {name} for {base}, saved to {file_out_path}")