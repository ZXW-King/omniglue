#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo script for performing OmniGlue inference."""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import src.omniglue as omniglue
from file import Walk
from src.omniglue import utils
from PIL import Image


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--imgLdir", type=str, default="", help="")
    parser.add_argument("--imgRdir", type=str, default="", help="")
    parser.add_argument("--topk", type=int, default=500, help="")
    args = parser.parse_args()
    return args


def run(filesL, filesR,save_path="",topk=1024):
    key = 32
    is_opencv_show = False
    for imgL, imgR in zip(filesL, filesR):
        if not os.path.exists(imgL) or not os.path.exists(imgR):
            raise ValueError(f"Image filepath '{imgL}' or '{imgR}' doesn't exist.")
        image0, image1, match_kp0, match_kp1, num_filtered_matches = match_img(imgL, imgR,topk=topk)
        img = show(imgL,imgR,image0, image1, match_kp0, match_kp1, num_filtered_matches)
        if save_path:
            res_img_name = "o_"+imgL.split("/")[-1].split(".")[0] + "__" + imgR.split("/")[-1].split(".")[
                0] + ".png"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img_file = os.path.join(save_path,res_img_name)
            if is_opencv_show:
                cv2.imwrite(img_file,img)
            else:
                img.save_plot(img_file)
        if is_opencv_show:
            cv2.imshow('res', img)
            if key == 27:
                print('Quitting, \'q\' pressed.')
                break
            if key == 32:
                while True:
                    key = cv2.waitKey(1)
                    if key == 32:
                        break
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    print('==> Finshed Test.')


def match_img(imgL, imgR,topk):
    # Load images.
    print("> Loading images...")
    image0 = cv2.imread(imgL)
    image1 = cv2.imread(imgR)

    # Load models.
    print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
    start = time.time()
    og = omniglue.OmniGlue(
        og_export="./models/og_export",
        sp_export="./models/sp_v6",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
        topk=topk
    )
    print(f"> \tTook {time.time() - start} seconds.")

    # Perform inference.
    print("> Finding matches...")
    start = time.time()
    match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
    num_matches = match_kp0.shape[0]
    print(f"> \tFound {num_matches} matches.")
    print(f"> \tTook {time.time() - start} seconds.")

    # Filter by confidence (0.02).
    print("> Filtering matches...")
    match_threshold = 0.02  # Choose any value [0.0, 1.0).
    keep_idx = []
    for i in range(match_kp0.shape[0]):
        if match_confidences[i] > match_threshold:
            keep_idx.append(i)
    num_filtered_matches = len(keep_idx)
    match_kp0 = match_kp0[keep_idx]
    match_kp1 = match_kp1[keep_idx]
    match_confidences = match_confidences[keep_idx]
    print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")
    return image0, image1, match_kp0, match_kp1, num_filtered_matches


def show(image0_path,image1_path,image0, image1, match_kp0, match_kp1, num_filtered_matches):
    # Visualize.
    print("> Visualizing matches...")
    viz = utils.visualize_matches(
        image0_path,
        image1_path,
        image0,
        image1,
        match_kp0,
        match_kp1,
        np.eye(num_filtered_matches),
        show_keypoints=True,
        highlight_unmatched=True,
        title=f"{num_filtered_matches} matches",
        line_width=2,
    )

    # viz = viz[..., ::-1]
    return viz



if __name__ == "__main__":
    # 禁用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args = GetArgs()
    filesL = Walk(args.imgLdir, ['jpg', 'jpeg', 'png'])
    filesR = Walk(args.imgRdir, ['jpg', 'jpeg', 'png'])
    topk = args.topk
    run(filesL,filesR,topk=topk)
