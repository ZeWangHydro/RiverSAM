import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import tifffile

def process_image(image_path, mode):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class Model(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(Model, self).__init__()
            self.unet = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=in_dim, classes=out_dim)

        def forward(self, input):
            y = self.unet(input)
            return F.leaky_relu(y, 0.2)

    UNet = Model(3, 2)
    model = UNet.to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    output = F.softmax(model(image_bgr), dim=1)
    output = output[:, 1, :, :].cpu().numpy()

    CHECKPOINT_PATH = os.path.join("sam_vit_h_4b8939.pth")

    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_predictor = SamPredictor(sam)

    if mode == "point prompt":
        points = np.argmax(output)
        points = points.reshape(-1, 2)
        labels = [[1]]

        mask_predictor.set_image(image_rgb)

        mask_predictor.predict(point_coords=points, point_labels=labels)
        masks, scores, logits = mask_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        tifffile.imwrite(os.path.splitext(image_path)[0] + ".tif", masks[np.argmax(scores)])

    elif mode == "everything mode":
        sam_result = mask_generator.generate(image_rgb)

        max_value = np.max(output)
        max_value_indices = np.argmax(output)
        max_value_position = tuple(max_value_indices[0])
        seg_number = len(sam_result)

        max_area = 0
        max_area_mask = None
        for j in range(0, seg_number):
            mask = sam_result[j]['segmentation']

            if mask[max_value_position[0], max_value_position[1]]:
                area = np.count_nonzero(mask)
                if area > max_area:
                    max_area = area
                    max_area_mask = mask

        if np.count_nonzero(max_area_mask):
            max_area_mask[max_area_mask == True] = 1
            max_area_mask[max_area_mask == False] = 0
            tifffile.imwrite(os.path.splitext(image_path)[0] + ".tif", max_area_mask)

# 示例调用
image_path = "xxxx.jpg"
mode = "point prompt" or "everything mode"
process_image(image_path, mode)
