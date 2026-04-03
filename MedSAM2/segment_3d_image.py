
import argparse
import os
from os.path import join
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import blosc2
import pickle

from sam2.build_sam import build_sam2_video_predictor_npz

# Set precision and seeds for reproducibility
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def read_nnunet_case(folder, case_id):
    """
    Read a case from the nnU-Net preprocessed format (.b2nd and .pkl).
    """
    # File paths
    b2nd_file = join(folder, f"{case_id}.b2nd")
    pkl_file = join(folder, f"{case_id}.pkl")

    # Read the compressed numpy array
    img_array_comp = blosc2.open(b2nd_file)
    # The actual image data is usually in the first channel
    img_array = img_array_comp[0, :, :, :]

    # Read the metadata
    with open(pkl_file, 'rb') as f:
        metadata = pickle.load(f)

    # Reconstruct a SimpleITK image object from metadata to keep a consistent workflow
    sitk_img = sitk.GetImageFromArray(img_array)
    sitk_stuff = metadata.get('sitk_stuff', {})
    sitk_img.SetSpacing(sitk_stuff.get('spacing', [1.0, 1.0, 1.0]))
    sitk_img.SetOrigin(sitk_stuff.get('origin', [0.0, 0.0, 0.0]))
    sitk_img.SetDirection(sitk_stuff.get('direction', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))

    return sitk_img

def preprocess_ct(image_data, window_level=-750, window_width=1500):
    """
    Apply windowing and normalization for CT images.
    """
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    return image_data_pre

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    
    return resized_array

@torch.inference_mode()
def run_segmentation(args):
    """
    Main function to run 3D image segmentation.
    """
    print("Initializing predictor...")
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    
    print(f"Reading case '{args.case_id}' from nnU-Net folder: {args.nnunet_folder}")
    sitk_img = read_nnunet_case(args.nnunet_folder, args.case_id)
    img_3d = sitk.GetArrayFromImage(sitk_img)
    
    print("Preprocessing image...")
    img_3d_preprocessed = preprocess_ct(img_3d)

    # Resize and normalize the image for the model
    video_height = img_3d_preprocessed.shape[1]
    video_width = img_3d_preprocessed.shape[2]
    img_resized = resize_grayscale_to_rgb_and_resize(img_3d_preprocessed, 1024)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    
    # Standard normalization
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std

    # Initialize the segmentation mask
    final_seg = np.zeros(img_3d.shape, dtype=np.uint8)

    print("Starting segmentation process for each point prompt...")
    for obj_id, point_str in enumerate(tqdm(args.coords, desc="Processing Prompts"), 1):
        coords = [int(c) for c in point_str.split(',')]
        if len(coords) != 3:
            print(f"Skipping invalid coordinate format: {point_str}")
            continue
        
        voxel_x, voxel_y, voxel_z = coords[0], coords[1], coords[2]

        if not (0 <= voxel_z < img_3d.shape[0] and 0 <= voxel_y < img_3d.shape[1] and 0 <= voxel_x < img_3d.shape[2]):
            print(f"Skipping out-of-bounds coordinates: {point_str}")
            continue

        point_coords = np.array([[voxel_x, voxel_y]], dtype=np.float32)
        point_labels = np.array([1], np.int32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            
            _, _, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=voxel_z,
                obj_id=obj_id,
                points=point_coords,
                labels=point_labels,
            )
            
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=voxel_z, reverse=False):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                final_seg[out_frame_idx, mask] = obj_id

            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            _, _, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=voxel_z,
                obj_id=obj_id,
                points=point_coords,
                labels=point_labels,
            )
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=voxel_z, reverse=True):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                final_seg[out_frame_idx, mask] = np.maximum(final_seg[out_frame_idx, mask], (mask * obj_id).astype(np.uint8))

    print(f"Saving final segmentation to: {args.output_path}")
    output_sitk = sitk.GetImageFromArray(final_seg)
    output_sitk.CopyInformation(sitk_img)
    sitk.WriteImage(output_sitk, args.output_path)
    
    print("Segmentation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a 3D medical image from nnU-Net format using MedSAM2 with point prompts.")
    
    parser.add_argument(
        "--nnunet_folder",
        type=str,
        required=True,
        help="Path to the nnU-Net preprocessed data folder (e.g., 'nnUNetPlans_3d_fullres').",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        required=True,
        help="The case identifier to process (e.g., 'case000').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output segmentation mask file.",
    )
    parser.add_argument(
        "--coords",
        nargs='+',
        required=True,
        help="List of 3D coordinates for point prompts, formatted as 'x,y,z'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/MedSAM2_latest.pt",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="Path to the model config file.",
    )

    args = parser.parse_args()
    
    b2nd_file = join(args.nnunet_folder, f"{args.case_id}.b2nd")
    if not os.path.exists(b2nd_file):
        print(f"Error: Input .b2nd file not found at {b2nd_file}")
    else:
        run_segmentation(args)
