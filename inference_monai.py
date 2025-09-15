from importlib.metadata import metadata
import os
import glob
import json
from ssl import ALERT_DESCRIPTION_CERTIFICATE_EXPIRED
import numpy as np
from typing import Any, Dict, List, Optional, Union
from monai.apps.auto3dseg import EnsembleRunner, AlgoEnsembleBestByFold
from monai.bundle import ConfigParser
from monai.transforms import MeanEnsemble, SaveImage
from monai.utils.enums import AlgoKeys
from monai.utils.module import look_up_option, optional_import
from monai.utils import ensure_tuple
from monai.data import FolderLayout, MetaTensor
from monai.transforms import Transform
import warnings
from monai.apps.auto3dseg.utils import get_name_from_algo_id, import_bundle_algo_history
from copy import deepcopy
from warnings import warn
import SimpleITK as sitk


def prepare_prob_for_saving(
    prob_array: np.ndarray,
    meta_data: dict,
    channel_dim: int | None = 0,
    squeeze_end_dims: bool = True,
) -> np.ndarray:
    # 检查元数据中的空间形状
    meta_spatial_shape = ensure_tuple(meta_data.get("spatial_shape", ()))
    
    # 自动关闭 channel_dim（如果元数据形状 >= 数据形状）
    if len(meta_spatial_shape) >= len(prob_array.shape):
        channel_dim = None  # 视为无通道数据
    elif channel_dim is None:
        warnings.warn(
            f"数据形状 {prob_array.shape} (空间形状 {meta_spatial_shape}) "
            f"但 channel_dim=None（可能丢失通道信息）"
        )
    
    # 处理压缩末尾维度
    if squeeze_end_dims:
        prob_array = np.squeeze(prob_array)
        if channel_dim is not None and prob_array.ndim > len(meta_spatial_shape):
            # 将通道维度移到末尾并压缩单维度
            prob_array = np.moveaxis(prob_array, channel_dim, -1)
            prob_array = prob_array[(..., 0) if prob_array.shape[-1] == 1 else ...]
    
    return prob_array

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
    # 5. Define custom ensemble class for probability maps
class ProbAlgoEnsemble(AlgoEnsembleBestByFold):
    def ensemble_pred_softmax(self, preds, sigmoid=True):
        """Return probability maps without argmax"""
        if any(not p.is_cuda for p in preds):
            preds = [p.cpu() for p in preds]
        return MeanEnsemble()(preds)  # Returns [C, H, W, D] probability tensor
    
    def __call__(self, pred_param: dict | None = None) -> list:
        param = {} if pred_param is None else deepcopy(pred_param)
        files = self.infer_files
        
        if "infer_files" in param:
            files = param.pop("infer_files")
        if "files_slices" in param:
            slices = param.pop("files_slices")
            files = files[slices]
        if "mode" in param:
            mode = param.pop("mode")
            self.mode = look_up_option(mode, supported=["mean", "vote"])
        
        sigmoid = param.pop("sigmoid", False)
        if "image_save_func" in param:
            img_saver = ConfigParser(param["image_save_func"]).get_parsed_content()
        
        algo_spec_params = param.pop("algo_spec_params", {})
        outputs = []
        
        for _, file in (
            enumerate(tqdm(files, desc="Ensembling (rank 0)..."))
            if has_tqdm and pred_param and pred_param.get("rank", 0) == 0
            else enumerate(files)
        ):
            preds = []
            for algo in self.algo_ensemble:
                infer_algo_name = get_name_from_algo_id(algo[AlgoKeys.ID])
                infer_instance = algo[AlgoKeys.ALGO]
                _param = self._apply_algo_specific_param(algo_spec_params, param, infer_algo_name)
                pred = infer_instance.predict(predict_files=[file], predict_params=_param)
                preds.append(pred[0])
            
            if "image_save_func" in param:
                try:
                    ensemble_preds = self.ensemble_pred(preds, sigmoid=sigmoid)
                    ensemble_probs = self.ensemble_pred_softmax(preds, sigmoid=sigmoid)
                except BaseException:
                    ensemble_preds = self.ensemble_pred([_.to("cpu") for _ in preds], sigmoid=sigmoid)
                    ensemble_probs = self.ensemble_pred_softmax([_.to("cpu") for _ in preds], sigmoid=sigmoid)
                
                res_prob = img_saver(ensemble_probs)
                if hasattr(res_prob, "meta") and "saved_to" in res_prob.meta.keys():
                    res_prob = res_prob.meta["saved_to"]
                    prob_path = res_prob.replace(".nii.gz", ".npz")
                    res_prob = sitk.GetArrayFromImage(sitk.ReadImage(res_prob))
                    np.savez_compressed(
                        prob_path, 
                        prob=res_prob
                    )
                res = img_saver(ensemble_preds)
                
                if hasattr(res, "meta") and "saved_to" in res.meta.keys():
                    res = res.meta["saved_to"]
                else:
                    warn("Image save path not returned.")
                    res = None
            else:
                warn("Prediction returned in list instead of disk, provide image_save_func to avoid out of memory.")
                res = self.ensemble_pred(preds, sigmoid=sigmoid)
            
            outputs.append(res)
        return outputs

# 6. Wrapper for EnsembleRunner
class WrappedEnsembleRunner(EnsembleRunner):
    def set_ensemble_method(self, ensemble_method_name, **kwargs: Any) -> None:
        if isinstance(ensemble_method_name, str):
            return super().set_ensemble_method(ensemble_method_name, **kwargs)
        else:
            self.ensemble_method = ensemble_method_name

def run_monai_inference_pipeline(
    work_dir: str,
    input_cfg: str,
    output_dir: str,
    test_data_dir: str,
    dataset_split_json: str,
    bundle_root_template: str = "/opt/app/resources/monai/output_dir/segresnet_{}",
    n_folds: int = 5,
    sigmoid: bool = False,
    resample: bool = False
) -> None:
    """
    Run MONAI inference pipeline with probability map saving capability.
    
    Args:
        work_dir: Working directory containing the model outputs
        input_cfg: Path to inference configuration YAML file
        output_dir: Directory to save inference results
        test_data_dir: Directory containing test data (NIfTI files)
        dataset_split_json: Path to dataset split JSON file
        bundle_root_template: Template path for bundle roots (should contain {} for fold number)
        n_folds: Number of cross-validation folds
        sigmoid: Whether to apply sigmoid activation to predictions
        resample: Whether to resample predictions to original image space
    """
    
    # 1. Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Update hyperparameters in all fold configs
    
    for i in range(n_folds):
        yaml_path = os.path.join(work_dir, f"segresnet_{i}", "configs", "hyper_parameters.yaml")
        new_values = {
            "bundle_root": os.path.join(work_dir, f"segresnet_{i}"),
            "data_list_file_path": dataset_split_json,
        }
        if not os.path.exists(yaml_path):
            print(f"Warning: {yaml_path} does not exist, skipping...")
            continue
        
        config = ConfigParser.load_config_file(yaml_path)
        for key, value in new_values.items():
            if key in config:
                config[key] = value
            else:
                print(f"Warning: Key '{key}' not found in {yaml_path}")
        
        ConfigParser.export_config_file(config, yaml_path, indent=2)
        print(f"Updated: {yaml_path}")
    
    print("Batch update completed!")
    
    # 3. Prepare test data
    files = sorted(glob.glob(os.path.join(test_data_dir, "*_0000.nii.gz")))
    secfiles = sorted(glob.glob(os.path.join(test_data_dir, "*_0001.nii.gz")))
    test_dict = [{"image": os.path.join(*f.split(os.sep)[-1:]), 
                  "image2": os.path.join(*f2.split(os.sep)[-1:])} 
                 for f, f2 in zip(files, secfiles)]
    
    # 4. Update datalist
    input_yaml = ConfigParser.load_config_file(input_cfg)
    print(input_yaml)
    data_list_path = input_yaml.get("datalist", "")
    
    if os.path.exists(data_list_path):
        with open(data_list_path, 'r') as f:
            data = json.load(f)
        data["testing"] = test_dict
        with open(data_list_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated testing data in {data_list_path}")
    else:
        raise FileNotFoundError(f"Datalist not found at {data_list_path}")
     
    # 7. Run inference
    runner = WrappedEnsembleRunner(
        data_src_cfg_name=input_cfg,
        work_dir=work_dir,
        mgpu=False,
        output_dir=output_dir,
        ensemble_method_name=ProbAlgoEnsemble(n_fold=n_folds),

        sigmoid=sigmoid,
        output_dtype="float32",
        output_postfix="prob",
        resample=resample
    )
    runner.device_setting = {
            "CUDA_VISIBLE_DEVICES": "0",
            "n_devices": 1,
            "NUM_NODES": 1,
            "MN_START_METHOD": os.environ.get("MN_START_METHOD", "bcprun"),
            "CMD_PREFIX": os.environ.get("CMD_PREFIX", ""),
    }
    print(f"Starting inference with multi-GPU support: {runner.mgpu}")
    runner.run()