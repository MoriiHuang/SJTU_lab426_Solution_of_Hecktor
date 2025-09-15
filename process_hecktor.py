import nibabel as nib
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union
import SimpleITK as sitk
import subprocess
from typing import Union, Optional, Tuple, List
import numpy as np
from skimage import measure
import shutil
import os

# 固定窗宽窗位参数
FIXED_CT_WINDOW = {'level': 45, 'width': 300}  # 头颈部软组织窗
FIXED_PET_WINDOW = {'level': 2.5, 'width': 5, 'normalize': True}  # SUV标准化
def get_label_file(case_dir: Path, ct_file: Path, pet_file: Path) -> Optional[Path]:
    """智能获取标签文件：排除CT/PET后剩下的.nii.gz文件"""
    # 获取目录下所有nii.gz文件
    all_files = list(case_dir.glob('*.nii.gz'))
    
    # 排除已知的CT和PET文件
    candidate_files = [
        f for f in all_files 
        if f not in (ct_file, pet_file) 
        and not f.name.startswith(('CT_', 'PT_'))
    ]
    
    # 如果有多个候选文件，选择体积最小的（通常标签文件较小）
    if len(candidate_files) == 1:
        return candidate_files[0]
    elif len(candidate_files) > 1:
        # 按文件大小排序，取最小的
        return sorted(candidate_files, key=lambda x: x.stat().st_size)[0]
    return None
def crop_head_neck_region(data: np.ndarray, z_threshold: int = 300) -> np.ndarray:
    """自动裁剪头颈区域（若z>300则视为全身扫描）"""
    if data.shape[2] <= z_threshold:
        return data
    # 经验值：头颈部约160层（1mm层厚）或总高度的30%
    print(f"Cropping data from {data.shape[2]} to 160 layers for head/neck region.")
    crop_z = min(160, int(data.shape[2] * 0.3))
    return data[..., -crop_z:]

def apply_window(data: np.ndarray, level: float, width: float, normalize: bool = False) -> np.ndarray:
    """应用窗宽窗位处理"""
    if normalize:
        data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    window_min = level - width / 2
    window_max = level + width / 2
    windowed = (data - window_min) / (window_max - window_min + 1e-6)
    return np.clip(windowed, 0, 1)

def apply_transforms(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对图像数据应用三种变换：
    1. 直接归一化 (Normalize)
    2. 像素平方映射 (x^2)
    3. 像素立方根映射 (x^(1/3))
    
    返回字典：{
        'normalized': 归一化结果,
        'squared': 平方映射结果,
        'cube_root': 立方根映射结果
    }
    """
    # 确保数据在[0,1]范围内
    normalized = (data - data.min()) / (data.max() - data.min() + 1e-6)
    
    return {
        'normalized': normalized,
        'squared': np.power(normalized, 2),
        'cube_root': np.power(normalized, 1/3)
    }

def save_as_nnunet_extended_sitk(
    ct_transforms: Dict[str, np.ndarray],
    pet_transforms: Dict[str, np.ndarray],
    reference_image: sitk.Image,
    case_id: str,
    output_dir: Union[str, Path],
    label: Optional[np.ndarray] = None
) -> None:
    """
    Save extended 6-channel data (CT_0000-0002, PET_0003-0005) using SimpleITK.
    
    Args:
        ct_transforms: Dictionary containing CT transforms ('normalized', 'squared', 'cube_root')
        pet_transforms: Dictionary containing PET transforms ('normalized', 'squared', 'cube_root')
        reference_image: SimpleITK image used for spatial reference (spacing, origin, direction)
        case_id: Case identifier string
        output_dir: Output directory path
        label: Optional label array to save
    """
    output_dir = Path(output_dir)
    
    # Create output directories if they don't exist
    (output_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)
    if label is not None:
        (output_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)
    
    def _save_image(array: np.ndarray, filename: Path) -> None:
        """Helper function to save numpy array as SimpleITK image"""
        img = sitk.GetImageFromArray(array.astype(np.float32))
        img.CopyInformation(reference_image)  # Copy metadata from reference
        sitk.WriteImage(img, str(filename))
    
    def _save_label(array: np.ndarray, filename: Path) -> None:
        """Helper function to save label array"""
        img = sitk.GetImageFromArray(array.astype(np.uint8))
        img.CopyInformation(reference_image)
        sitk.WriteImage(img, str(filename))
    
    # Save CT transforms
    _save_image(ct_transforms['normalized'], output_dir / 'imagesTr' / f'{case_id}_0000.nii.gz')
    _save_image(ct_transforms['squared'], output_dir / 'imagesTr' / f'{case_id}_0001.nii.gz')
    _save_image(ct_transforms['cube_root'], output_dir / 'imagesTr' / f'{case_id}_0002.nii.gz')
    
    # Save PET transforms
    _save_image(pet_transforms['normalized'], output_dir / 'imagesTr' / f'{case_id}_0003.nii.gz')
    _save_image(pet_transforms['squared'], output_dir / 'imagesTr' / f'{case_id}_0004.nii.gz')
    _save_image(pet_transforms['cube_root'], output_dir / 'imagesTr' / f'{case_id}_0005.nii.gz')
    
    # Save label if provided
    if label is not None:
        _save_label(label, output_dir / 'labelsTr' / f'{case_id}.nii.gz')

def save_as_nnunet_extended_sitk_inference(
    ct_transforms: Dict[str, np.ndarray],
    pet_transforms: Dict[str, np.ndarray],
    reference_image: sitk.Image,
    case_id: str,
    output_dir: Union[str, Path],
    label: Optional[np.ndarray] = None
) -> None:
    """
    Save extended 6-channel data (CT_0000-0002, PET_0003-0005) using SimpleITK.
    
    Args:
        ct_transforms: Dictionary containing CT transforms ('normalized', 'squared', 'cube_root')
        pet_transforms: Dictionary containing PET transforms ('normalized', 'squared', 'cube_root')
        reference_image: SimpleITK image used for spatial reference (spacing, origin, direction)
        case_id: Case identifier string
        output_dir: Output directory path
        label: Optional label array to save
    """
    output_dir = Path(output_dir)
    
    # Create output directories if they don't exist
    (output_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)
    if label is not None:
        (output_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)
    
    def _save_image(array: np.ndarray, filename: Path) -> None:
        """Helper function to save numpy array as SimpleITK image"""
        img = sitk.GetImageFromArray(array.astype(np.float32))
        img.CopyInformation(reference_image)  # Copy metadata from reference
        sitk.WriteImage(img, str(filename))
    
    def _save_label(array: np.ndarray, filename: Path) -> None:
        """Helper function to save label array"""
        img = sitk.GetImageFromArray(array.astype(np.uint8))
        img.CopyInformation(reference_image)
        sitk.WriteImage(img, str(filename))
    
    # Save CT transforms
    _save_image(ct_transforms['normalized'], output_dir /  f'{case_id}_0000.nii.gz')    
    # Save PET transforms
    _save_image(pet_transforms['normalized'], output_dir /  f'{case_id}_0001.nii.gz')    
    # Save label if provided
    if label is not None:
        _save_label(label, output_dir / f'{case_id}.nii.gz')

def resample_label_roi(image: sitk.Image, centroid: List[float],
                      output_size: List[int] = [144, 144, 144],
                      output_spacing: List[float] = [2, 2, 2]) -> sitk.Image:
    """专用标签重采样函数（使用最近邻插值）"""
    output_origin = [
        centroid[0] - output_size[0]/2 * output_spacing[0],
        centroid[1] - output_size[1]/2 * output_spacing[1],
        centroid[2] - output_size[2] * output_spacing[2],
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 关键区别
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

def resample_head_neck_region(
    sitk_img: sitk.Image,
    output_size = [256, 256, 128],
    output_spacing = [2, 2, 3],
    type: str = 'CT'
):
    """通过重采样裁剪头颈区域，并返回裁剪参数（用于后续还原）"""
    img_arr = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)
    original_spacing = sitk_img.GetSpacing()
    original_origin = sitk_img.GetOrigin()
    original_size = sitk_img.GetSize()

    # 计算裁剪后的原点（从头部开始取128层）
    highest_z = original_origin[2] + original_spacing[2] * img_arr.shape[0]
    output_origin_z = highest_z - output_size[2] * output_spacing[2]
    output_origin = [original_origin[0], original_origin[1], output_origin_z]

    # 重采样裁剪
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1)
    cropped_img = resampler.Execute(sitk_img)
    # 保存裁剪参数（用于还原）
    crop_params = {
        "original_spacing": original_spacing,
        "original_origin": original_origin,
        "original_size": original_size,
        "output_origin": output_origin,
        "output_spacing": output_spacing,
        "output_size": output_size,
    }
    return cropped_img, crop_params

def run_nnunet_prediction(input_dir: str, output_dir: str,
                         model: str = "Dataset004_HECKROI",
                         folds: str = "all",
                         checkpoint = "checkpoint_best.pth",
                         save_probabilities=False,
                         plan = None) -> str:
    """Run nnUNet prediction using both CT and PET modalities"""
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", model,
        "-c", "3d_fullres",
        "-npp","1",
        "-nps", "1",
        "--disable_tta"
    ]
    if folds:
        cmd.append('-f')
        # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
        fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
        cmd.extend(fold_list)
    if checkpoint:
        cmd.append('-chk')
        cmd.append(str(checkpoint))
    if plan:
        cmd.append('-p')
        cmd.append(str(plan))
    if save_probabilities:
        cmd.append('--save_probabilities')
    
    cmd_str = [str(item) if isinstance(item, Path) else item for item in cmd]
    print(f"Running nnUNet prediction with command: {' '.join(cmd_str)}")
    subprocess.run(cmd_str, check=True)
    # Return the first prediction file found (assuming single prediction per case)
    return next(Path(output_dir).glob("*.nii.gz"), None)


def get_head_centroid(pred_mask_path: str) -> Tuple[List[float], List[float], List[float]]:
    """Calculate centroid from predicted head mask"""
    mask_img = sitk.ReadImage(pred_mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    
    # Find largest connected component
    all_labels = measure.label(mask_arr, background=0, connectivity=1)
    properties = measure.regionprops(all_labels)
    if not properties:
        raise ValueError("No head region found in prediction")
    
    largest_region = max(properties, key=lambda x: x.area)
    bbox = largest_region.bbox  # z,y,x
    
    # Calculate centroid coordinates (weighted toward superior part)
    y = (bbox[1] + bbox[4]) // 2
    x = (bbox[2] + bbox[5]) // 2
    z = int(bbox[0] + (bbox[3] - bbox[0]) * 0.7)
    
    # Convert to physical coordinates
    real_coord_xyz = [
        (x - 1) * spacing[0] + origin[0],
        (y - 1) * spacing[1] + origin[1],
        (z - 1) * spacing[2] + origin[2],
    ]
    return real_coord_xyz, spacing, origin

def resample_roi(image: sitk.Image, centroid: List[float],
                 output_size: List[int] = [144, 144, 144],
                 output_spacing: List[float] = [2, 2, 2],type = 'CT') -> sitk.Image:
    """Resample ROI around centroid"""
    output_origin = [
        centroid[0] - output_size[0]/2 * output_spacing[0],
        centroid[1] - output_size[1]/2 * output_spacing[1],
        centroid[2] - output_size[2] * output_spacing[2],
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(sitk.sitkBSpline)
    if type == 'CT':
        resampler.SetDefaultPixelValue(-1)
    else:
        resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

def get_tumor_centroid(label_path: str) -> Tuple[List[float], List[float], List[float]]:
    """Calculate centroid from tumor label mask"""
    label_img = sitk.ReadImage(label_path)
    label_arr = sitk.GetArrayFromImage(label_img)
    spacing = label_img.GetSpacing()
    origin = label_img.GetOrigin()
    
    # Get coordinates of all tumor voxels (label 1 or 2)
    coords = np.where(label_arr != 0)
    
    if len(coords[0]) == 0:
        raise ValueError("No tumor region found in label")
    
    # Calculate mean centroid coordinates (z,y,x)
    center = [
        np.mean(coords[0]),  # z
        np.mean(coords[1]),  # y
        np.mean(coords[2]),  # x
    ]
    
    # Convert to physical coordinates (x,y,z)
    real_coord_xyz = [
        center[2] * spacing[0] + origin[0],
        center[1] * spacing[1] + origin[1],
        center[0] * spacing[2] + origin[2],
    ]
    return real_coord_xyz, spacing, origin

def resample_tumor_roi(image:sitk.Image, centroid: List[float], 
                      output_size: List[int] = [144, 144, 144],
                      output_spacing: List[float] = [1, 1, 1],
                      img_type: str = 'CT') -> sitk.Image:
    """Resample ROI around tumor centroid"""
    # Calculate output origin centered on tumor
    output_origin = [
        centroid[0] - output_size[0]/2 * output_spacing[0],
        centroid[1] - output_size[1]/2 * output_spacing[1],
        centroid[2] - output_size[2]/2 * output_spacing[2],
    ]
    
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetSize(output_size)
    
    if img_type == 'label':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(-1 if img_type == 'CT' else 0)
    
    return resampler.Execute(image)

def inference_process(ct_file: str,
                      pet_file: str,
                      temp_dir: str,
                      task_head: str = "Dataset004_HECKHead",
                      task_coarse: str = "Dataset005_HECKCoarse",
                      task_fine: str = "Dataset014_HECKFine2chan",
                      out_path: str = "output") -> None:
    try:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_head_localization_dir = temp_dir / 'head_localization'
        temp_head_localization_dir.mkdir(parents=True, exist_ok=True)
        case_id = Path(ct_file).stem.split('_')[0]
        ct_image = sitk.ReadImage(ct_file)
        pet_image = sitk.ReadImage(pet_file)
        ct_cropped, ct_crop_params = resample_head_neck_region(ct_image,type='CT')
        pet_cropped, pet_crop_params = resample_head_neck_region(pet_image, type='PET')
        

        # 转换为numpy数组供后续处理
        ct_data = sitk.GetArrayFromImage(ct_cropped)
        pet_data = sitk.GetArrayFromImage(pet_cropped)

        ct_data = apply_window(ct_data, 
                                FIXED_CT_WINDOW['level'], 
                                FIXED_CT_WINDOW['width'])
        pet_data = apply_window(pet_data, 
                                FIXED_PET_WINDOW['level'], 
                                FIXED_PET_WINDOW['width'], 
                                FIXED_PET_WINDOW['normalize'])

        ct_image = sitk.GetImageFromArray(ct_data)
        ct_image.CopyInformation(ct_cropped)  # 保留原始信息
        pet_image = sitk.GetImageFromArray(pet_data)
        pet_image.CopyInformation(pet_cropped)  # 保留原始信息
        temp_ct = temp_dir / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(ct_image, str(temp_ct))
        pred_mask_path = run_nnunet_prediction(temp_dir,temp_head_localization_dir,task_head)
        if not pred_mask_path:
            raise ValueError("No prediction generated")

        # Step 2: Get head centroid from prediction
        centroid, _, _ = get_head_centroid(str(pred_mask_path))
        print(f"Head centroid for {case_id}: {centroid}")
        res_ct  = resample_roi(ct_image, centroid,type='CT')
        res_pet = resample_roi(pet_image, centroid,type='PET')
        temp_coarse_input = temp_dir / 'coarse_input'
        temp_coarse_input.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(res_ct, str(temp_coarse_input / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(res_pet, str(temp_coarse_input / f"{case_id}_0001.nii.gz"))
        # Step 3: Run coarse prediction
        coarse_output_dir = temp_dir / 'coarse_output'
        coarse_output_dir.mkdir(parents=True, exist_ok=True)
        pred_mask = run_nnunet_prediction(str(temp_coarse_input), str(coarse_output_dir), task_coarse, folds='all')
        if not pred_mask:
            raise ValueError("No coarse prediction generated")
        # Step 4: Get tumor centroid from coarse prediction
        tumor_centroid, _, _ = get_tumor_centroid(str(pred_mask))
        print(f"Tumor centroid for {case_id}: {tumor_centroid}")
        res_ct = resample_tumor_roi(ct_image, tumor_centroid, img_type='CT')
        res_pet = resample_tumor_roi(pet_image, tumor_centroid, img_type='PET')
        ct_transforms = apply_transforms(sitk.GetArrayFromImage(res_ct))
        pet_transforms = apply_transforms(sitk.GetArrayFromImage(res_pet))
        temp_fine_input = temp_dir / 'fine_input'
        temp_fine_input.mkdir(parents=True, exist_ok=True)
        save_as_nnunet_extended_sitk_inference(
            ct_transforms=ct_transforms,
            pet_transforms=pet_transforms,
            reference_image=res_ct,
            case_id=case_id,
            output_dir=temp_fine_input
        )
        
        # Step 5: Run fine prediction
        fine_output_dir = temp_dir / 'fine_output'
        fine_output_dir.mkdir(parents=True, exist_ok=True)
        fine_mask=run_nnunet_prediction(str(temp_fine_input), str(fine_output_dir), task_fine, folds='0,1,2,3,4',save_probabilities=True,plan="nnUNetResEncUNetLPlans")
        print("output dir has the following files:", os.listdir(fine_output_dir))
        # # step 5: resample fine mask to original size
        # fine_mask_sitk = sitk.ReadImage(fine_mask)
        # fine_mask_resampled = sitk.ResampleImageFilter()
        # origin_ct_image =sitk.ReadImage(ct_file)
        # output_size = origin_ct_image.GetSize()
        # fine_mask_resampled.SetSize(output_size)
        # fine_mask_resampled.SetOutputSpacing(origin_ct_image.GetSpacing())
        # fine_mask_resampled.SetOutputOrigin(origin_ct_image.GetOrigin())
        # fine_mask_resampled.SetOutputDirection(origin_ct_image.GetDirection())
        # fine_mask_resampled.SetInterpolator(sitk.sitkNearestNeighbor)
        # fine_mask_resampled.SetDefaultPixelValue(0)
        # fine_mask_resampled = fine_mask_resampled.Execute(fine_mask_sitk)
        # # 保存结果
        # sitk.WriteImage(fine_mask_resampled, str(out_path))    
    finally:
        # # 清理临时目录
        # if temp_dir.exists():
        #     shutil.rmtree(temp_dir)
        print(f"Processed {case_id}: Results saved to {out_path}")

def process(input_dir: Union[str, Path], output_dir: Union[str, Path],
             preview_dir: Optional[Union[str, Path]] = None,
             save_cut = False,
             save_croi = False,
             save_froi = False) -> None:
    """主处理函数（已集成裁剪和窗宽窗位选择）"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    (output_dir/'imagesTr').mkdir(parents=True, exist_ok=True)
    (output_dir/'labelsTr').mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
    
    for case_dir in sorted(input_dir.glob('*')):
        if not case_dir.is_dir():
            continue
            
        try:
            temp_dir = Path(output_dir / 'temp')
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_head_localization_dir = temp_dir / 'head_localization'
            temp_head_localization_dir.mkdir(parents=True, exist_ok=True)
            case_id = case_dir.name.split('_')[0]
            ct_file = next(case_dir.glob('*__CT.nii.gz'), None)
            pet_file = next(case_dir.glob('*__PT.nii.gz'), None)
            
            if not ct_file or not pet_file:
                print(f"Skipping {case_dir.name}: missing CT/PET")
                continue
                
            # 加载数据
            label = get_label_file(case_dir, ct_file, pet_file)
            ct_sitk = sitk.ReadImage(str(ct_file))
            pet_sitk = sitk.ReadImage(str(pet_file))
            ct_cropped, ct_crop_params = resample_head_neck_region(ct_sitk,type='CT')
            pet_cropped, pet_crop_params = resample_head_neck_region(pet_sitk, type='PET')
            
        
            # 转换为numpy数组供后续处理
            ct_data = sitk.GetArrayFromImage(ct_cropped)
            pet_data = sitk.GetArrayFromImage(pet_cropped)

            ct_data = apply_window(ct_data, 
                                   FIXED_CT_WINDOW['level'], 
                                   FIXED_CT_WINDOW['width'])
            pet_data = apply_window(pet_data, 
                                    FIXED_PET_WINDOW['level'], 
                                    FIXED_PET_WINDOW['width'], 
                                    FIXED_PET_WINDOW['normalize'])
        
            ct_image = sitk.GetImageFromArray(ct_data)
            ct_image.CopyInformation(ct_cropped)  # 保留原始信息
            pet_image = sitk.GetImageFromArray(pet_data)
            pet_image.CopyInformation(pet_cropped)  # 保留原始信息

            if save_cut:
                # save CT and PET images
                ct_output_path = output_dir / 'imagesTr' / f"{case_id}_0000.nii.gz"
                pet_output_path = output_dir / 'imagesTr' / f"{case_id}_0001.nii.gz"
                sitk.WriteImage(ct_image, str(ct_output_path))
                sitk.WriteImage(pet_image, str(pet_output_path))
                print(f"Processed {case_id}: CT and PET saved.")
            if save_croi:
                temp_ct = temp_dir / f"{case_id}_0000.nii.gz"
                sitk.WriteImage(ct_image, str(temp_ct))
                pred_mask_path = run_nnunet_prediction(temp_dir,temp_head_localization_dir)
                if not pred_mask_path:
                    raise ValueError("No prediction generated")
        
                # Step 2: Get head centroid from prediction
                centroid, _, _ = get_head_centroid(str(pred_mask_path))
                print(f"Head centroid for {case_id}: {centroid}")
                res_ct  = resample_roi(ct_image, centroid,type='CT')
                res_pet = resample_roi(pet_image, centroid,type='PET')
                if label:
                    label_sitk = sitk.ReadImage(str(label))
                    label_resampled = resample_label_roi(label_sitk, centroid)
                    label_resampled = sitk.Cast(label_resampled, sitk.sitkUInt8)
                    label_resampled.CopyInformation(res_ct)  # 保留原始信息
                    sitk.WriteImage(label_resampled, str(output_dir / 'labelsTr' / f"{case_id}.nii.gz"))
                    print(f"Processed {case_id}: Resampled label saved.")
                sitk.WriteImage(res_ct, str(output_dir / 'imagesTr' / f"{case_id}_0000.nii.gz"))
                sitk.WriteImage(res_pet, str(output_dir / 'imagesTr' / f"{case_id}_0001.nii.gz"))
                print(f"Processed {case_id}: Cropped CT and PET saved.")
            if save_froi:
                if label:
                    label_path = str(label)
                    centroid, _, _ = get_tumor_centroid(label_path)
                else:
                    tumor_path = run_nnunet_prediction(temp_dir, temp_head_localization_dir)
                    centroid, _, _ = get_head_centroid(str(tumor_path))
                print(f"Tumor centroid for {case_id}: {centroid}")
                res_ct = resample_tumor_roi(ct_image, centroid, img_type='CT')
                res_pet = resample_tumor_roi(pet_image, centroid, img_type='PET')

                # sitk.WriteImage(res_ct, str(output_dir / 'imagesTr' / f"{case_id}_0000.nii.gz"))
                # sitk.WriteImage(res_pet, str(output_dir / 'imagesTr' / f"{case_id}_0001.nii.gz"))
                ct_transforms = apply_transforms(sitk.GetArrayFromImage(res_ct))
                pet_transforms = apply_transforms(sitk.GetArrayFromImage(res_pet))
                save_as_nnunet_extended_sitk(
                    ct_transforms,
                    pet_transforms,
                    res_ct,
                    case_id,
                    output_dir
                )
                if label:
                    label_sitk = sitk.ReadImage(str(label))
                    label_resampled = resample_tumor_roi(label_sitk, centroid, img_type='label')
                    label_resampled.CopyInformation(res_ct)
                    sitk.WriteImage(label_resampled, str(output_dir / 'labelsTr' / f"{case_id}.nii.gz"))
                    print(f"Processed {case_id}: Resampled label ROI saved.")
                print(f"Processed {case_id}: Resampled tumor ROI saved.")


        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
if __name__ == '__main__':
    # 配置路径
    input_dir = "/home/SSD-2T-2023/hcy/nnUNet_new/DATASET/nnUNet_raw/Task1"
    # input_dir = "/home/SSD-2T-2023/hcy/nnUNet_new/DATASET/nnUNet_raw/Task_test"
    output_dir = "/home/SSD-2T-2023/hcy/nnUNet_new/DATASET/nnUNet_raw/Dataset006_HECKFine"
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    # 执行处理
    process(
        input_dir=input_dir,
        output_dir=output_dir,
        save_cut=False,
        save_croi= False,
        save_froi=True,
    )
    
