# Copyright 2025 Diagnostic Image Analysis Group, Radboud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, Tuple
import pandas as pd
from collections import defaultdict
# function resample_img is copied from: https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py
class CenterFingerprint:
    """中心指纹统计与管理类：提取图像分布特征，并记录平均窗宽窗位"""
    def __init__(self):
        self.center_stats = defaultdict(list)
        self.final_stats = {}

    def extract_features(self, img: np.ndarray, prefix: str) -> dict:
        """提取多种图像分布特征"""
        img = img[img > np.percentile(img, 80)]
        return {
            f'{prefix}_mean': float(img.mean()),
            f'{prefix}_std': float(img.std()),
            f'{prefix}_min': float(img.min()),
            f'{prefix}_max': float(img.max()),
            f'{prefix}_percentile25': float(np.percentile(img, 25)),
            f'{prefix}_percentile50': float(np.percentile(img, 50)),
            f'{prefix}_percentile95': float(np.percentile(img, 75)),
        }

    def add_case(self, center: str, ct_img: np.ndarray, pet_img: np.ndarray, ct_params: dict, pet_params: dict):
        """添加图像统计特征 + 平均窗宽窗位"""
        features = {}
        features.update(self.extract_features(ct_img, 'ct'))
        features.update(self.extract_features(pet_img, 'pet'))
        features.update({
            'ct_level': ct_params['level'],
            'ct_width': ct_params['width'],
            'pet_level': pet_params['level'],
            'pet_width': pet_params['width'],
        })
        self.center_stats[center].append(features)

    def compute_final_stats(self):
        """计算各中心指纹 + 平均窗宽窗位"""
        for center, records in self.center_stats.items():
            df = pd.DataFrame(records)
            summary = {col: float(df[col].mean()) for col in df.columns}
            summary['sample_count'] = int(len(df))
            self.final_stats[center] = summary
        return self.final_stats

    def save_to_json(self, filepath: Union[str, Path]):
        if not self.final_stats:
            self.compute_final_stats()
        with open(filepath, 'w') as f:
            json.dump(self.final_stats, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]):
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls()
        instance.final_stats = data
        return instance
def match_center_by_features(fingerprint_data: dict, sample_features: dict) -> Tuple[str, dict]:
    """通过图像分布特征匹配最相似中心（欧氏距离）"""
    min_dist, best_center = float('inf'), None
    target_keys = [k for k in sample_features.keys() if k in list(next(iter(fingerprint_data.values())).keys())]
    x = np.array([sample_features[k] for k in target_keys])
    for center, stats in fingerprint_data.items():
        y = np.array([stats[k] for k in target_keys])
        dist = np.linalg.norm(x - y)
        if dist < min_dist:
            min_dist, best_center = dist, center
    return best_center, fingerprint_data[best_center]

def extract_features_from_sample(ct_img_path: Union[str, Path], pet_img_path: Union[str, Path]) -> dict:
    """给定新样本图像路径，提取其图像特征用于匹配"""
    ct_img = nib.load(str(ct_img_path)).get_fdata()
    pet_img = nib.load(str(pet_img_path)).get_fdata()
    extractor = CenterFingerprint()
    features = {}
    features.update(extractor.extract_features(ct_img, 'ct'))
    features.update(extractor.extract_features(pet_img, 'pet'))
    return features
from nibabel.processing import resample_from_to
def resize_to_match(ct_img: nib.Nifti1Image, 
                   pet_img: nib.Nifti1Image) -> tuple:
    # Ensure both images are 3D
    if len(ct_img.shape) != 3 or len(pet_img.shape) != 3:
        raise ValueError("Both images must be 3D.")
    
    # Resample PET image to match CT image's shape and affine matrix
    resampled_pet = resample_from_to(pet_img, (ct_img.shape, ct_img.affine), order=1)
    
    return (ct_img, resampled_pet)
def calculate_overlap(pancreas_mask, component_mask):
	"""
	Calculate the overlap between a pancreas mask and a component mask.
	
	Args:
		pancreas_mask (numpy.ndarray): Binary mask of the pancreas
		component_mask (numpy.ndarray): Binary mask of the connected component
		
	Returns:
		tuple: (overlap_voxels, overlap_percentage)
			- overlap_voxels: Number of overlapping voxels
			- overlap_percentage: Percentage of pancreas covered by the component
	"""
	import numpy as np
	
	# Ensure both are binary masks
	pancreas_mask = pancreas_mask > 0
	component_mask = component_mask > 0
	
	# Calculate overlap (intersection)
	overlap_voxels = np.sum(pancreas_mask & component_mask)
	
	# Calculate percentage of pancreas covered by the component
	pancreas_voxels = np.sum(pancreas_mask)
	component_voxels = np.sum(component_mask)
	if component_voxels == 0:
		overlap_percentage = 0.0
	else:
		overlap_percentage = 100.0 * overlap_voxels / component_voxels
	
	return overlap_voxels, overlap_percentage

def filter_components(labels, num_features, pancreas_mask, min_overlap_percent=15.0, min_voxels=10):
	"""
	Filter connected components based on minimum overlap with pancreas mask and minimum size.
	
	Args:
		labels (numpy.ndarray): Array with labeled connected components
		num_features (int): Number of connected components 
		pancreas_mask (numpy.ndarray): Binary mask of the pancreas
		min_overlap_percent (float): Minimum overlap percentage threshold
		min_voxels (int): Minimum number of voxels a component must have
		
	Returns:
		numpy.ndarray: Binary mask with only components meeting the threshold
	"""
	import numpy as np
	
	# Create an empty mask for the filtered result
	filtered_mask = np.zeros_like(labels, dtype=bool)
	
	# Track which components meet the criteria
	kept_components = []
	
	for i in range(1, num_features+1):
		component = (labels == i)
		component_voxels = np.sum(component)
		overlap_voxels, overlap_percentage = calculate_overlap(pancreas_mask, component)
		
		# Check both conditions: minimum overlap percentage and minimum size
		if overlap_percentage >= min_overlap_percent and component_voxels >= min_voxels:
			# Add this component to our filtered mask
			filtered_mask = filtered_mask | component
			kept_components.append(i)
	
	print(f"Kept {len(kept_components)}/{num_features} components: {kept_components}")
	return filtered_mask

def resample_img(itk_image, out_spacing  = [3.0, 3.0, 6.0], is_label=False, out_size = [], out_origin = [], out_direction= []):
	"""
	Resamples an ITK image to a specified voxel spacing, optionally adjusting its size, origin, and direction.

	This function modifies the spatial resolution of a given medical image by changing its voxel spacing. 
	It can be used for both intensity images (e.g., CT, MRI) and segmentation masks, using appropriate interpolation methods.

	Parameters:
	-----------
	itk_image : sitk.Image
		The input image in SimpleITK format.
	
	out_spacing : list of float, optional (default: [2.0, 2.0, 2.0])
		The desired voxel spacing in (x, y, z) directions (in mm).
	
	is_label : bool, optional (default: False)
		Whether the input image is a label/segmentation mask.
		- `False`: Uses B-Spline interpolation for smooth intensity images.
		- `True`: Uses Nearest-Neighbor interpolation to preserve label values.
	
	out_size : list of int, optional (default: [])
		The desired output image size (in voxels). If not provided, it is automatically computed 
		to preserve the original physical image dimensions.
	
	out_origin : list of float, optional (default: [])
		The desired output image origin (in physical space). If not provided, the original image origin is used.
	
	out_direction : list of float, optional (default: [])
		The desired output image orientation. If not provided, the original image direction is used.

	Returns:
	--------
	itk_image : sitk.Image
		The resampled image with the specified voxel spacing, size, origin, and direction.

	Notes:
	------
	- The function ensures that the physical space of the image is preserved when resampling.
	- If `out_size` is not specified, it is automatically computed based on the original and target spacing.
	- If resampling a segmentation mask (`is_label=True`), nearest-neighbor interpolation is used to avoid label mixing.

	Example:
	--------
	```python
	# Resample an MRI image to 1mm isotropic resolution
	resampled_img = resample_img(mri_image, out_spacing=[1.0, 1.0, 1.0])

	# Resample a segmentation mask (preserving labels)
	resampled_mask = resample_img(segmentation_mask, out_spacing=[1.0, 1.0, 1.0], is_label=True)
	```
	"""
	import SimpleITK as sitk
	import numpy as np
	original_spacing = itk_image.GetSpacing()
	original_size    = itk_image.GetSize()
	

	if not out_size:
		out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
						int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
						int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
	
	# set up resampler
	resample = sitk.ResampleImageFilter()
	resample.SetOutputSpacing(out_spacing)
	resample.SetSize(out_size)
	if not out_direction:
		out_direction = itk_image.GetDirection()
	resample.SetOutputDirection(out_direction)
	if not out_origin:
		out_origin = itk_image.GetOrigin()
	resample.SetOutputOrigin(out_origin)
	resample.SetTransform(sitk.Transform())
	resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
	if is_label:
		resample.SetInterpolator(sitk.sitkNearestNeighbor)
	else:
		resample.SetInterpolator(sitk.sitkBSpline)
	# perform resampling
	itk_image = resample.Execute(itk_image)

	return itk_image

def upsample_mask(low_res_mask, source_spacing, target_spacing, fill_holes=True):
	"""
	Upsample a low-resolution mask using spacing information.
	
	Parameters:
	-----------
	low_res_mask : numpy.ndarray
		The segmentation mask predicted at low resolution.
	source_spacing : tuple or list of float
		Spacing of the low-resolution mask (e.g., (z, y, x)).
	target_spacing : tuple or list of float
		Spacing of the target image (e.g., (z, y, x)).
	fill_holes : bool, optional
		Whether to fill holes in the upsampled mask.
		
	Returns:
	--------
	upsampled_mask : numpy.ndarray
		The mask upsampled to the target spacing.
	"""
	import numpy as np
	from scipy import ndimage

	# Compute zoom factor for each axis: ratio of source_spacing to target_spacing.
	zoom_factors = [s / t for s, t in zip(source_spacing, target_spacing)]
	
	# Upsample using nearest neighbor interpolation (order=0).
	upsampled_mask = ndimage.zoom(low_res_mask, zoom=zoom_factors, order=0)
	
	if fill_holes:
		# If binary segmentation, fill holes slice by slice.
		# For multi-class segmentation, you'd need to iterate over each label.
		upsampled_mask = ndimage.binary_fill_holes(upsampled_mask).astype(upsampled_mask.dtype)
	
	return upsampled_mask