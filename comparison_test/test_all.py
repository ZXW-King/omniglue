import os
import sys
from pathlib import Path

from lightglue_test.onnx_infer import run_infer
from test import run

class TestCompared:
    def __init__(self,json_file,extractor_type):
        self.json_file = json_file
        self.extractor_type = extractor_type

    def test_lightglue(self):
        extractor_path = f"lightglue_test/weights/{self.extractor_type}.onnx"
        image_lists,save_res,valid_K = run_infer(json_file, self.extractor_type, extractor_path)
        return image_lists,save_res,valid_K

    def test_omniglue(self,filesL, filesR,save_path,topk):
        run(filesL, filesR,save_path,topk)

    def run_all(self):
        image_lists, save_res, topk = self.test_lightglue()
        print("lightglue 测试完成！")
        imgL, imgR = zip(*image_lists)
        self.test_omniglue(list(imgL),list(imgR),save_res,topk)
        print("omniglue 测试完成！")



if __name__ == '__main__':
    # 禁用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 获取当前文件所在的目录
    current_dir = Path(__file__).resolve().parent
    # 假设项目目录是当前文件所在目录的父目录
    project_dir = current_dir.parent
    # 切换到项目目录
    os.chdir(project_dir)
    # 将项目目录添加到 sys.path
    sys.path.append(str(project_dir))
    json_file = "lightglue_test/data/config/test_easy_500_topK.json"
    extractor_type = "superpoint"
    test = TestCompared(json_file,extractor_type)
    test.run_all()


