import os
import sys
from pathlib import Path

from lightglue_test.infer import infer
import json





def run_infer(json_file,extractor_type,extractor_path):
    with open(json_file, 'r') as f_json:
        configs = json.load(f_json)
        image_cate = configs["image_cate"]
        valid_K = configs["valid_K"]
        strategy = configs["strategy"]
        model_type = configs["model_type"]
        num_layer = configs["num_layer"]
        lightglue_path = configs["lightglue_path"]
        image_lists = configs["image_lists"]
        npu_matches_dir = configs["npu_matches"]
        onnx_matches_dir = configs["onnx_matches"]
        save_res = configs["res_save_path"]

    for image_pairs in image_lists:
        m_kpts0, m_kpts1 = infer(
            image_cate=image_cate,
            valid_K=valid_K,
            strategy=strategy,
            model_type=model_type,
            num_layer=num_layer,
            npu_matches_dir=npu_matches_dir,
            onnx_matches_dir=onnx_matches_dir,
            img_paths=image_pairs,
            extractor_type=extractor_type,
            extractor_path=extractor_path,
            lightglue_path=lightglue_path,
            img_size=512,
            viz=True,
            save_path=save_res
        )
    return image_lists,save_res,valid_K


if __name__ == '__main__':
    # 获取当前文件所在的目录
    current_dir = Path(__file__).resolve().parent
    # 假设项目目录是当前文件所在目录的父目录
    project_dir = current_dir.parent
    # 切换到项目目录
    os.chdir(project_dir)
    # 将项目目录添加到 sys.path
    sys.path.append(str(project_dir))
    json_file = "lightglue_test/data/config/test_darker_uchar_500_forefront.json"
    extractor_type = "superpoint"  # "disk"
    extractor_path = f"lightglue_test/weights/{extractor_type}.onnx"
    run_infer(json_file,extractor_type,extractor_path)


