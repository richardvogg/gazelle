import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import torch
from torch.utils.data import Dataset

from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

#lemurs_smallnew_ViTB14/2025-07-29_15-02-09/epoch_19.pt

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="./data/B_e10_c7/B_e10_c7.mp4")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument("--ckpt_path", type=str, default="./experiments/lemurs_smallnew_ViTB14_largeinputsquare/2025-08-05_19-10-03/epoch_19.pt") #"./experiments/train_gazelle_vitb_lemurs/2025-06-17_18-09-57/epoch_7.pt"
parser.add_argument("--output_path", type=str, default="./output_images/B_e10_c7_largeinputsquare/")
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()



import cv2
import torch
from torch.utils.data import Dataset

class VideoFrameDatasetWithBBoxes(Dataset):
    def __init__(self, video_path, transform=None, image_width=1920, image_height=1080):
        self.video_path = video_path
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height

        # Count frames
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()

        # Load annotations
        self.bboxes_per_frame = self._load_annotations()

    def _load_annotations(self):
        bboxes = {}
        with open(self.video_path.replace(".mp4", ".txt"), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                obj_id = int(parts[1])
                cls = int(parts[-1])
                if cls != 0:
                    continue

                x, y, w, h = map(float, parts[2:6])
                x1 = x / self.image_width
                y1 = y / self.image_height
                x2 = (x + w) / self.image_width
                y2 = (y + h) / self.image_height

                bbox = [obj_id, x1, y1, x2, y2]
                bboxes.setdefault(frame_idx, []).append(bbox)
        return bboxes

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IndexError(f"Failed to read frame {idx}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        bboxes = self.bboxes_per_frame.get(idx)

        frame_number = idx
        return frame, frame_number, bboxes

    
def collate(batch):
    images, frame_number, bboxes = zip(*batch)
    return torch.stack(images), list(frame_number), list(bboxes)


@torch.no_grad()
def main(vis = False):
    if vis:
        os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()
    # Create a list of indices starting at idx=2000 and skipping every 5th idx
    indices = list(range(1000, len(VideoFrameDatasetWithBBoxes(args.video_path, transform)), 5))

    dataset = VideoFrameDatasetWithBBoxes(args.video_path, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        sampler=indices
    )

    output_width = 1080
    output_height = 608

    for k, (images, frame_number, bboxes) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        detections_present = bboxes[0] is not None
        if detections_present:
            # Remove obj_id from each bbox: keep only coordinates
            bboxes_clean = [
                [[*bbox[1:]] for bbox in frame_bboxes] if frame_bboxes is not None else None
                for frame_bboxes in bboxes
            ]
            preds = model.forward({"images": images.to(device), "bboxes": bboxes_clean})

        # eval each instance (head)
        for i in range(images.shape[0]): # per image

            if vis:
                hm_alpha = 0.5  # Change this for stronger or weaker heatmap visibility

                # Convert image tensor [3, 448, 448] to RGB numpy array
                img_tensor = images[i].cpu()

                img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                img_np = (img_np - np.min(img_np)) / (np.ptp(img_np) + 1e-8)  # Normalize to [0,1]
                img_np = cv2.resize(img_np, (output_width, output_height))  # Resize to match output dimensions

                # Plot the image with the bounding boxes drawn on top
                # Set DPI and figure size for high quality output
                figsize = (12, 10)  # Size in inches

                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(img_np)

            if detections_present:
                for j in range(len(bboxes[i])): # per individual


                    # Resize heatmap [1, 64, 64] -> [1, 896, 896]
                    heatmap_tensor = preds['heatmap'][i][j].unsqueeze(0).unsqueeze(0)  # shape: [1,1,64,64]
                    heatmap_resized = F.interpolate(heatmap_tensor, size=(896, 896), mode='bilinear', align_corners=False)
                    heatmap_resized = heatmap_resized.squeeze().cpu().numpy()  # shape: [896, 896]

                    # Normalize heatmap to [0, 1]
                    #heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.ptp(heatmap_resized) + 1e-8)
                    heatmap_norm = cv2.resize(heatmap_resized, (output_width, output_height))  # Resize to match image dimensions
                    if vis:
                        # Create alpha mask
                        alpha_mask = heatmap_norm.copy()
                        alpha_mask[alpha_mask < 0.05] = 0  # Fully transparent where heat is low
                        # Scale the remaining values to [0, hm_alpha]
                        alpha_mask = np.clip(alpha_mask, 0, 1) * hm_alpha   
                        alpha_mask = cv2.resize(alpha_mask, (output_width, output_height))  # Resize to match image dimensions

                        ax.imshow(heatmap_norm, cmap='jet', alpha=alpha_mask)  # Overlay heatmap

                    obj_id = bboxes[i][j][0]  # Get the object ID
                    x1, y1, x2, y2 = bboxes[i][j][1:]
                    if vis:
                        rect = plt.Rectangle((x1 * output_width, y1 * output_height), (x2 - x1) * output_width, (y2 - y1) * output_height, linewidth=2, edgecolor='yellow', facecolor='none')

                    # Plot the center of the rectangle
                    center_x = (x1 + x2) / 2 * output_width
                    center_y = (y1 + y2) / 2 * output_height
                    if vis:
                        ax.plot(center_x, center_y, 'yo', markersize=10)

                    if preds['inout'][i][j].item() > 0.2:
                        # Find the maximum of heatmap_norm and plot a yellow dot
                        max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
                        # Check if (max_x, max_y) is inside any other bbox (not the current one)
                        inside_other_bbox = False
                        target_id = None
                        for k, other_bbox in enumerate(bboxes[i]):
                            if k == j:
                                continue  # skip current bbox
                            ox1, oy1, ox2, oy2 = other_bbox[1:]
                            # Scale bbox to image size
                            ox1_img, oy1_img = ox1 * output_width, oy1 * output_height
                            ox2_img, oy2_img = ox2 * output_width, oy2 * output_height
                            if ox1_img <= max_x <= ox2_img and oy1_img <= max_y <= oy2_img:
                                inside_other_bbox = True
                                target_id = bboxes[i][k][0]  # Get the object ID of the other bbox
                                with open(os.path.join(args.output_path, "interaction_log.txt"), "a") as f:
                                    f.write(f"{frame_number[0]},{obj_id},{target_id},{preds['inout'][i][j].item()}\n")
                        if vis:
                            ax.plot(max_x, max_y, 'ro', markersize=10)  
                            # Draw a line connecting the center of the rectangle to the max heatmap location
                            ax.plot([center_x, max_x], [center_y, max_y], color='yellow', linewidth=2)

                    #elif preds['inout'][i][j].item() > 0.2:
                    #    # Find the maximum of heatmap_norm and plot a yellow dot
                    #    max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
                    #    ax.plot(max_x, max_y, 'bo', markersize=10)  

                    if vis:
                        ax.add_patch(rect)
                    

            if vis:
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(
                    f"{args.output_path}/frame_{frame_number[0]}.png",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi = 300
                )
                plt.close()

                
        
if __name__ == "__main__":
    main(vis = False)