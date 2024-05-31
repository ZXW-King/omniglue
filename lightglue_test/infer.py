import argparse
import os.path
from typing import List

from .onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale, viz2d
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_paths",
        nargs=2,
        required=True,
        type=str,
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        required=True,
        help="Path to the LightGlue ONNX model or end-to-end LightGlue pipeline.",
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        choices=["superpoint", "disk"],
        required=True,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=False,
        help="Path to the feature extractor ONNX model. If this argument is not provided, it is assumed that lightglue_path refers to an end-to-end model.",
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the images to this value. Otherwise, please provide two integers (height width) to resize both images to this size, or four integers (height width height width).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT (experimental).",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Whether to visualize the results."
    )
    return parser.parse_args()


def infer(
        image_cate, valid_K, strategy, model_type, num_layer,npu_matches_dir,onnx_matches_dir,
        img_paths: List[str],
        lightglue_path: str,
        extractor_type: str,
        extractor_path=None,
        img_size=512,
        trt=False,
        viz=False,
        save_path = ""
):
    # Handle args
    img0_path = img_paths[0]
    img1_path = img_paths[1]
    if isinstance(img_size, List):
        if len(img_size) == 1:
            size0 = size1 = img_size[0]
        elif len(img_size) == 2:
            size0 = size1 = img_size
        elif len(img_size) == 4:
            size0, size1 = img_size[:2], img_size[2:]
        else:
            raise ValueError("Invalid img_size. Please provide 1, 2, or 4 integers.")
    else:
        size0 = size1 = img_size

    image0, scales0 = load_image(img0_path, resize=size0)

    image1, scales1 = load_image(img1_path, resize=size1)

    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
    elif extractor_type == "disk":
        pass
    else:
        raise NotImplementedError(
            f"Unsupported feature extractor type: {extractor_type}."
        )

    # Load ONNX models
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if trt:
        providers = [
                        (
                            "TensorrtExecutionProvider",
                            {
                                "trt_fp16_enable": True,
                                "trt_engine_cache_enable": True,
                                "trt_engine_cache_path": "weights/cache",
                            },
                        )
                    ] + providers

    runner = LightGlueRunner(
        image_cate,
        valid_K,
        strategy,
        model_type,
        num_layer,
        npu_matches_dir=npu_matches_dir,
        onnx_matches_dir=onnx_matches_dir,
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
        providers=providers,
    )

    # Run inference
    m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1, img0_path, img1_path)

    # Visualisation
    if viz:
        print(" >>>>> img0_path = {0} - img1_path = {1} <<<<<< ".format(img0_path, img1_path))
        orig_image0, _ = load_image(img0_path)
        orig_image1, _ = load_image(img1_path)
        viz2d.plot_images(
            [orig_image0[0].transpose(1, 2, 0), orig_image1[0].transpose(1, 2, 0)]
        )
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.plot_matches(m_kpts0, m_kpts1, lw=0.2)
        viz2d.add_text(0, f'Match:{len(m_kpts0)}', pos=(0.2, 0.9), fs=20, color='yellow', lcolor='blue', lwidth=3, ha='center',
                       va='center')
        res_img_name = "l_" + img0_path.split("/")[-1].split(".")[0] + "__" + img1_path.split("/")[-1].split(".")[0] + ".png"
        # output_dir = "save_res/outputs/{0}_{1}_{2}_{3}_num_layer_{4}/".format(image_cate, valid_K, strategy, model_type,
        #                                                                   num_layer)
        # print(" >>> output_dir = {} <<<<".format(output_dir))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # viz2d.save_plot(output_dir + res_img_name)
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            viz2d.save_plot(os.path.join(save_path , res_img_name))

        # viz2d.plt.show()

    return m_kpts0, m_kpts1


if __name__ == "__main__":
    args = parse_args()
    m_kpts0, m_kpts1 = infer(**vars(args))
    print(m_kpts0, m_kpts1)
