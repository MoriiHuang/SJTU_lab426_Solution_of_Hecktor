#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.



from pathlib import Path
import time
import glob
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import shutil
from scipy import ndimage
from data_utils import *
from process_hecktor import *
from inference_monai import  run_monai_inference_pipeline
from monai.apps.auto3dseg import EnsembleRunner, AlgoEnsembleBestByFold
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

def cleanup_temp_files(keep_dirs=None):
    """清理临时文件，只保留指定目录中的文件"""
    if keep_dirs is None:
        keep_dirs = ["/output"]
    
    keep_files = set()
    for dir_path in keep_dirs:
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    keep_files.add(os.path.join(root, file))
    
    temp_dirs = [
        "/tmp/nnunet/input",
        "/tmp/nnunet/output",
        "/tmp/monai/input",
        "/tmp/monai/output",
        "/tmp/temp_dir"
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in keep_files:
                        try:
                            os.remove(file_path)
                            print(f"Deleted temporary file: {file_path}")
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
def fuse_probabilities(prob_modelA, prob_modelB):
    """
    融合两个模型的概率图：
    - 将ModelA的label=2合并到背景(label=0)
    - 对label=1的概率取加权平均
    """
    # 加权平均
    weight_A = 0.5
    weight_B = 0.5
    fused_prob = np.zeros_like(prob_modelA)
    labels = prob_modelA.shape[0]  # 假设第一个维度是类别数
    for i in range(labels):
        fused_prob[i] = (weight_A * prob_modelA[i] + weight_B * prob_modelB[i])
    # 确保概率归一化
    fused_prob = fused_prob / np.sum(fused_prob, axis=0, keepdims=True)
    
    return fused_prob

import warnings
warnings.filterwarnings("ignore")
def extract_filenames(file_paths):
    filenames = []
    for file_path in file_paths:
        # 获取文件名（带扩展名）
        basename = os.path.basename(file_path)
        # 去除扩展名
        filename = os.path.splitext(basename)[0]
        filenames.append(filename)
    return filenames


class HecktorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir = Path("/tmp/nnunet/input")
        self.nnunet_output_dir = Path("/tmp/nnunet/output")
        self.monai_input_dir = Path("/tmp/monai/input")
        self.monai_output_dir = Path("/tmp/monai/output")
        self.nnunet_model_dir = Path("/opt/app/resources/nnUNet_results")
        self.monai_model_dir = Path("/opt/app/resources/monai/output_dir")
        # input / output paths for predictions-model
        folders_with_ct = [folder for folder in os.listdir("/input/images") if "ct" in folder.lower()]
        folders_with_pet =[folder for folder in os.listdir("/input/images") if 'pet' in folder.lower()]
        if len(folders_with_ct) == 1: 
            ct_ip_dir_name = folders_with_ct[0]
            print("Folder containing eval image", ct_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'ct', but found", len(folders_with_ct))
            ct_ip_dir_name = 'ct' #default value
        if len(folders_with_pet) == 1: 
            pet_ip_dir_name = folders_with_pet[0]
            print("Folder containing eval image", pet_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_ct))
            pet_ip_dir_name = 'pet' #default value
        
        
        self.ct_ip_dir = Path(f"/input/images/{ct_ip_dir_name}") #abdominal-t2-mri
        self.pet_ip_dir = Path(f"/input/images/{pet_ip_dir_name}")
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "tumor-lymph-node-segmentation"))
        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.monai_input_dir.mkdir(exist_ok=True, parents = True)
        self.monai_output_dir.mkdir(exist_ok=True, parents = True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        ct_mha_files = (glob.glob(os.path.join(self.ct_ip_dir, '*.mha'))+glob.glob(os.path.join(self.ct_ip_dir, '*.tif'))+glob.glob(os.path.join(self.ct_ip_dir, '*.tiff'))+glob.glob(os.path.join(self.ct_ip_dir, '*.nii.gz')))
        pet_mha_files = (glob.glob(os.path.join(self.pet_ip_dir, '*.mha'))+glob.glob(os.path.join(self.pet_ip_dir, '*.tif'))+glob.glob(os.path.join(self.pet_ip_dir, '*.tiff'))+glob.glob(os.path.join(self.pet_ip_dir, '*.nii.gz')))
        # Check if any .mha files were found
        if ct_mha_files:
            self.ct_image = ct_mha_files 
        else:
            print('No mha images found in input directory')
        if pet_mha_files:
            self.pet_image = pet_mha_files
        self.file_name = str(extract_filenames(ct_mha_files)[0].split('.')[0].split('_')[0])
        print(self.pet_image,self.ct_image,self.file_name)
        self.segmentation_mask = self.output_dir_seg_mask / f"{self.file_name}.mha"

    def run(self):
        """
        Load T1 MRI and generate segmentation of the tumor 
        """
        try:
            _show_torch_cuda_info()
            start_time = time.perf_counter()
            task = "HECKTOR"
            print(f"Running segmentation for task: {task}")
    
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)
            inference_process(str(self.ct_image[0]),
                              str(self.pet_image[0]),
                              temp_dir=str('/tmp/temp_dir'),
                              out_path=str(self.segmentation_mask))
            monai_test_folder = str('/tmp/temp_dir/fine_input')
            work_dir = str(self.monai_model_dir)
            input_cfg = str(self.monai_model_dir / "inference.yaml")
            output_dir = str(self.monai_output_dir)
            data_split_json= str(self.monai_model_dir / "dataset_split.json")
            run_monai_inference_pipeline(work_dir,input_cfg,output_dir,monai_test_folder,data_split_json)
            print(os.listdir(output_dir))
            nnunet_prob_path = f'/tmp/temp_dir/fine_output/{self.file_name}.npz'
            nnunet_mask_path = f'/tmp/temp_dir/fine_output/{self.file_name}.nii.gz'
            nnunet_mask = sitk.ReadImage(nnunet_mask_path)
            monai_prob_path= f'{output_dir}/{self.file_name}_0000_prob.npz'
            monai_mask_path= f'{output_dir}/{self.file_name}_0000_prob.nii.gz'
            nnunet_prob = np.load(nnunet_prob_path)['probabilities']
            monai_prob = np.load(monai_prob_path)['prob']
            print("Is equal:?",np.argmax(monai_prob,axis=0).all()== sitk.GetArrayFromImage(sitk.ReadImage(monai_mask_path)).all())
            fused_prob = fuse_probabilities(nnunet_prob,monai_prob)
            fused_mask_array = np.argmax(fused_prob,axis=0)
            fused_mask_sitk = sitk.GetImageFromArray(fused_mask_array)
            fused_mask_sitk.CopyInformation(nnunet_mask)
            # fused_mask_sitk = sitk.ReadImage(monai_mask_path)

            fine_mask_resampled = sitk.ResampleImageFilter()
            origin_ct_image =sitk.ReadImage(str(self.ct_image[0]))
            output_size = origin_ct_image.GetSize()
            fine_mask_resampled.SetSize(output_size)
            fine_mask_resampled.SetOutputSpacing(origin_ct_image.GetSpacing())
            fine_mask_resampled.SetOutputOrigin(origin_ct_image.GetOrigin())
            fine_mask_resampled.SetOutputDirection(origin_ct_image.GetDirection())
            fine_mask_resampled.SetInterpolator(sitk.sitkNearestNeighbor)
            fine_mask_resampled.SetDefaultPixelValue(0)
            fine_mask_resampled = fine_mask_resampled.Execute(fused_mask_sitk)
            # 保存结果
            print("处理前的数据类型：",fine_mask_resampled.GetPixelIDTypeAsString())
            fine_mask_resampled = sitk.Cast(fine_mask_resampled, sitk.sitkUInt8) 
            print("处理后的数据类型：",fine_mask_resampled.GetPixelIDTypeAsString())
            print("唯一标签值：", np.unique(sitk.GetArrayFromImage(fine_mask_resampled)))  # 应为 [0, 1, 2]
            sitk.WriteImage(fine_mask_resampled,str(self.segmentation_mask))
            end_time = time.perf_counter()
            print(f"Total processing time: {end_time - start_time:.3f} seconds")
        finally:
            print("Cleaning up temporary files...")
            cleanup_temp_files()
    def predict(self, input_dir, output_dir, task="Dataset091_PantherTask2", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2"):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
                fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
                cmd.extend(fold_list)

            if checkpoint:
                cmd.append('-chk')
                cmd.append(str(checkpoint))

            cmd_str = " ".join(cmd)
            print(f"Running command: {cmd_str}")
            subprocess.check_call(cmd_str, shell=True)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    HecktorSegmentationContainer().run()
