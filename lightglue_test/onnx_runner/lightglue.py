import os

import numpy as np
import onnxruntime as ort
import torch


class LightGlueRunner:
    def __init__(
            self,
            image_cate,
            valid_K,
            strategy,
            model_type,
            num_layer,
            lightglue_path,
            npu_matches_dir='',
            onnx_matches_dir='',
            extractor_path=None,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.image_cate = image_cate
        self.valid_K = valid_K
        self.strategy = strategy
        self.model_type = model_type
        self.num_layer = num_layer
        self.onnx_matches_dir = onnx_matches_dir
        self.npu_matches_dir = npu_matches_dir
        self.extractor = (
            ort.InferenceSession(
                extractor_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if extractor_path is not None
            else None
        )
        sess_options = ort.SessionOptions()
        self.lightglue = ort.InferenceSession(
            lightglue_path, sess_options=sess_options, providers=providers
        )

        # Check for invalid models.
        lightglue_inputs = [i.name for i in self.lightglue.get_inputs()]
        if self.extractor is not None and "image0" in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is end-to-end. Please do not pass the extractor_path argument."
            )
        elif self.extractor is None and "image0" not in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is not end-to-end. Please pass the extractor_path argument."
            )

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1, img0_path, img1_path,is_save_point=False):
        if self.extractor is None:
            kpts0, kpts1, matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "image0": image0,
                    "image1": image1,
                },
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1
        else:
            kpts0, scores0, desc0 = self.extractor.run(None, {"image": image0})
            kpts1, scores1, desc1 = self.extractor.run(None, {"image": image1})

            # matches0, mscores0 = self.lightglue_test.run(
            #     None,
            #     {
            #         "kpts0": self.normalize_keypoints(
            #             kpts0, image0.shape[2], image0.shape[3]
            #         ),
            #         "kpts1": self.normalize_keypoints(
            #             kpts1, image1.shape[2], image1.shape[3]
            #         ),
            #         "desc0": desc0,
            #         "desc1": desc1,
            #     },
            # )

            ## modify by jcw

            norm_kpts0 = self.normalize_keypoints(kpts0, image0.shape[2], image0.shape[3])
            print(" >>>>> h = {0} , w = {1} <<<<".format(image0.shape[2], image0.shape[3]))
            norm_kpts1 = self.normalize_keypoints(kpts1, image0.shape[2], image0.shape[3])

            print(" >>>>>> norm_kpts0.shape = {} <<<<<<".format(norm_kpts0.shape))
            print(" >>>>>> norm_kpts1.shape = {} <<<<<<".format(norm_kpts1.shape))

            norm_kpts_len = min(norm_kpts0.shape[1], norm_kpts1.shape[1])

            if self.strategy == "topK":

                score0_tensor = torch.Tensor(scores0)
                actual_len = score0_tensor.shape[1]

                if actual_len < self.valid_K:
                    score0_sorted = torch.topk(score0_tensor, actual_len)
                    topK_index = score0_sorted.indices[0][:actual_len]
                    last_index = torch.Tensor([topK_index[-1].tolist()] * (self.valid_K - actual_len))
                    topK_index = torch.concat([topK_index, last_index], dim=0)
                else:
                    score0_sorted = torch.topk(score0_tensor, self.valid_K)
                    topK_index = score0_sorted.indices[0][:self.valid_K]
                topK_index = topK_index.int()

                kpts0_topK = norm_kpts0[..., topK_index, :]
                desc0_topK = desc0[..., topK_index, :]

                score1_tensor = torch.Tensor(scores1)
                actual_len = score1_tensor.shape[1]
                if actual_len < self.valid_K:
                    score1_sorted = torch.topk(score1_tensor, actual_len)
                    topK_index = score1_sorted.indices[0][:actual_len]
                    last_index = torch.Tensor([topK_index[-1].tolist()] * (self.valid_K - actual_len))
                    topK_index = torch.concat([topK_index, last_index], dim=0)
                else:
                    score1_sorted = torch.topk(score1_tensor, self.valid_K)
                    topK_index = score1_sorted.indices[0][:self.valid_K]
                topK_index = topK_index.int()

                kpts1_topK = norm_kpts1[..., topK_index, :]
                desc1_topK = desc1[..., topK_index, :]

                kpts0_new = kpts0_topK
                kpts1_new = kpts1_topK
                desc0_new = desc0_topK
                desc1_new = desc1_topK

            elif self.strategy == "forefront":
                score0_tensor = torch.Tensor(scores0)
                actual_len = score0_tensor.shape[1]

                if actual_len < self.valid_K:
                    forefront_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([forefront_index[-1].tolist()] * (self.valid_K - actual_len))
                    forefront_index = torch.concat([forefront_index, last_index], dim=0)
                else:
                    forefront_index = torch.Tensor(np.arange(self.valid_K))
                forefront_index = forefront_index.int()

                kpts0_forefront = norm_kpts0[..., forefront_index, :]
                desc0_forefront = desc0[..., forefront_index, :]

                kpts0_origin = kpts0[..., forefront_index, :]
                desc0_origin = desc0[..., forefront_index, :]

                score1_tensor = torch.Tensor(scores1)
                actual_len = score1_tensor.shape[1]
                if actual_len < self.valid_K:
                    forefront_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([forefront_index[-1].tolist()] * (self.valid_K - actual_len))
                    forefront_index = torch.concat([forefront_index, last_index], dim=0)
                else:
                    forefront_index = torch.Tensor(np.arange(self.valid_K))
                forefront_index = forefront_index.int()

                kpts1_forefront = norm_kpts1[..., forefront_index, :]
                desc1_forefront = desc1[..., forefront_index, :]

                kpts1_origin = kpts1[..., forefront_index, :]
                desc1_origin = desc1[..., forefront_index, :]

                kpts0_new = kpts0_forefront
                kpts1_new = kpts1_forefront
                desc0_new = desc0_forefront
                desc1_new = desc1_forefront

            elif self.strategy == "back":
                score0_tensor = torch.Tensor(scores0)
                actual_len = score0_tensor.shape[1]
                print(" >>>>>  actual_len = {} <<<<<".format(actual_len))

                if actual_len < self.valid_K:
                    back_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([back_index[-1].tolist()] * (self.valid_K - actual_len))
                    back_index = torch.concat([back_index, last_index], dim=0)
                else:
                    back_index = torch.Tensor(np.arange(int(actual_len - self.valid_K), actual_len))

                back_index = back_index.int()
                # print(" >>>>> first >  back_index = {} <<<<<".format(back_index))

                kpts0_back = norm_kpts0[..., back_index, :]
                desc0_back = desc0[..., back_index, :]

                score1_tensor = torch.Tensor(scores1)
                actual_len = score1_tensor.shape[1]
                if actual_len < self.valid_K:
                    back_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([back_index[-1].tolist()] * (self.valid_K - actual_len))
                    back_index = torch.concat([back_index, last_index], dim=0)
                else:
                    back_index = torch.Tensor(np.arange(int(actual_len - self.valid_K), actual_len))
                back_index = back_index.int()
                # print(" >>>>> second > back_index = {} <<<<<".format(back_index))
                kpts1_back = norm_kpts1[..., back_index, :]
                desc1_back = desc1[..., back_index, :]

                kpts0_new = kpts0_back
                kpts1_new = kpts1_back
                desc0_new = desc0_back
                desc1_new = desc1_back


            elif self.strategy == "middle":
                score0_tensor = torch.Tensor(scores0)
                actual_len = score0_tensor.shape[1]

                if actual_len < self.valid_K:
                    middle_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([middle_index[-1].tolist()] * (self.valid_K - actual_len))
                    middle_index = torch.concat([middle_index, last_index], dim=0)
                else:
                    middle_index = torch.Tensor(np.arange(actual_len)[
                                                (int(actual_len / 2.0) - int(self.valid_K / 2.0)): (
                                                            int(actual_len / 2.0) + int(self.valid_K / 2.0))])
                middle_index = middle_index.int()
                print(" >>>>>> middle_index = {} <<<<<<".format(middle_index))

                kpts0_middle = norm_kpts0[..., middle_index, :]
                desc0_middle = desc0[..., middle_index, :]

                score1_tensor = torch.Tensor(scores1)
                actual_len = score1_tensor.shape[1]
                if actual_len < self.valid_K:
                    middle_index = torch.Tensor(np.arange(actual_len))
                    last_index = torch.Tensor([middle_index[-1].tolist()] * (self.valid_K - actual_len))
                    middle_index = torch.concat([middle_index, last_index], dim=0)
                else:
                    middle_index = torch.Tensor(np.arange(actual_len)[
                                                (int(actual_len / 2.0) - int(self.valid_K / 2.0)): (
                                                            int(actual_len / 2.0) + int(self.valid_K / 2.0))])
                middle_index = middle_index.int()

                kpts1_middle = norm_kpts0[..., middle_index, :]
                desc1_middle = desc0[..., middle_index, :]

                kpts0_new = kpts0_middle
                kpts1_new = kpts1_middle
                desc0_new = desc0_middle
                desc1_new = desc1_middle

            elif self.strategy == "random":
                pass
            elif self.strategy == "sequence":
                pass
            else:
                pass

            """

            print(" >>>>> actual_len = {} <<<<<".format(actual_len))

            print(" >>>>> lightglue_test.py > kpts0_new.shape = {} <<<<<<".format(kpts0_new.shape))
            # print( " >>>>> lightglue_test.py > kpts0_new = {} <<<<<<".format( kpts0_new ) )
            # print(" >>>>> lightglue_test.py > kpts1_new = {} <<<<<<".format(kpts1_new))
            # print(" >>>>> lightglue_test.py > desc0_new = {} <<<<<<".format(desc0_new))
            # print(" >>>>> lightglue_test.py > desc1_new = {} <<<<<<".format(desc1_new))

            import os
            if not os.path.exists("save_res/input_bins/{0}/".format(self.image_cate)):
                os.makedirs("save_res/input_bins/{0}/".format(self.image_cate))

            image_name = img0_path.split("/")[-1].split(".")[0] + "__" + img1_path.split("/")[-1].split(".")[0]
            np.array(kpts0_new).tofile(
                "save_res/input_bins/{0}/{1}_topK_kpts0.bin".format(self.image_cate, image_name))
            np.array(desc0_new).tofile(
                "save_res/input_bins/{0}/{1}_topK_desc0.bin".format(self.image_cate, image_name))
            np.array(kpts1_new).tofile(
                "save_res/input_bins/{0}/{1}_topK_kpts1.bin".format(self.image_cate, image_name))
            np.array(desc1_new).tofile(
                "save_res/input_bins/{0}/{1}_topK_desc1.bin".format(self.image_cate, image_name))

            import os
            if not os.path.exists("save_res/input_bins_origin/{0}/".format(self.image_cate)):
                os.makedirs("save_res/input_bins_origin/{0}/".format(self.image_cate))

            # print( " >>>>>> kpts0_origin.shape = {} <<<<<".format( kpts0_origin.shape ) )
            # print(" >>>>>> kpts1_origin.shape = {} <<<<<".format(kpts1_origin.shape))
            # print(" >>>>>> desc0_origin.shape = {} <<<<<".format(desc0_origin.shape))
            # print(" >>>>>> desc1_origin.shape = {} <<<<<".format(desc1_origin.shape))
            # print(" >>>>>> kpts0_origin = {} <<<<<".format(kpts0_origin))
            # print(" >>>>>> kpts1_origin = {} <<<<<".format(kpts1_origin))

            image_name = img0_path.split("/")[-1].split(".")[0] + "__" + img1_path.split("/")[-1].split(".")[0]
            np.array(kpts0_origin).astype(float).tofile(
                "save_res/input_bins_origin/{0}/{1}_topK_kpts0.bin".format(self.image_cate, image_name))
            np.array(desc0_origin).astype(float).tofile(
                "save_res/input_bins_origin/{0}/{1}_topK_desc0.bin".format(self.image_cate, image_name))
            np.array(kpts1_origin).astype(float).tofile(
                "save_res/input_bins_origin/{0}/{1}_topK_kpts1.bin".format(self.image_cate, image_name))
            np.array(desc1_origin).astype(float).tofile(
                "save_res/input_bins_origin/{0}/{1}_topK_desc1.bin".format(self.image_cate, image_name))

            print(" >>>>> kpts0_origin.shape = {} <<<<".format(kpts0_origin.shape))
            print(" >>>>> type(kpts0_origin) = {} <<<<".format(type(kpts0_origin)))

            np.savetxt(
                "save_res/input_bins_origin/{0}/{1}_topK_kpts0.txt".format(self.image_cate, image_name),
                kpts0_origin.squeeze(), fmt='%s', delimiter='\t')
            np.savetxt(
                "save_res/input_bins_origin/{0}/{1}_topK_desc0.txt".format(self.image_cate, image_name),
                desc0_origin.squeeze(), fmt='%s', delimiter='\t')
            np.savetxt(
                "save_res/input_bins_origin/{0}/{1}_topK_kpts1.txt".format(self.image_cate, image_name),
                kpts1_origin.squeeze(), fmt='%s', delimiter='\t')
            np.savetxt(
                "save_res/input_bins_origin/{0}/{1}_topK_desc1.txt".format(self.image_cate, image_name),
                desc1_origin.squeeze(), fmt='%s', delimiter='\t')

            #### if float2uchar
            print(" >>>>> img0_path = {} <<<<".format(img0_path))
            print(" >>>>> img1_path = {} <<<<".format(img1_path))
            img0_name = img0_path.split("/")[-1].split(".")[0]
            img1_name = img1_path.split("/")[-1].split(".")[0]
            prefix_name = "{0}__{1}".format(img0_name, img1_name)
            this_kpts0 = np.fromfile(
                "save_res/input_bins/{}/{}_topK_kpts0.bin".format(self.image_cate,prefix_name),
                "float32").reshape([1, 500, 2])
            this_kpts1 = np.fromfile(
                "save_res/input_bins/{}/{}_topK_kpts1.bin".format(self.image_cate,prefix_name),
                "float32").reshape([1, 500, 2])
            this_desc0 = np.loadtxt(
                "save_res/input_bins/{}/{}_topK_desc0.txt".format(self.image_cate,prefix_name)).reshape(
                [1, 500, 256])
            this_desc1 = np.loadtxt(
                "save_res/input_bins/{}/{}_topK_desc1.txt".format(self.image_cate,prefix_name)).reshape(
                [1, 500, 256])

            # norm_desc0 = np.loadtxt("save_res/input_bins/{}/{}_topK_desc0_norm.txt".format(prefix_name),delimiter=',').reshape([1,500, 256])
            # print(" >>>>> norm_desc0.shape = {} <<<<<<".format(norm_desc0.shape))
            # norm_desc1 = np.loadtxt("save_res/input_bins/{}/{}_topK_desc1_norm.txt".format(prefix_name),delimiter=',').reshape([1, 500, 256])

            print(" >>>>> this_desc0.shape = {} <<<<<<".format(this_desc0.shape))
            print(" >>>>> this_kpts0.shape = {} <<<<<<".format(this_kpts0.shape))

            print(" >>>>> type(this_desc0[0][0][0]) = {} <<<<<<".format(type(this_desc0[0][0][0])))
            print(" >>>>> type(this_kpts0[0][0][0]) = {} <<<<<<".format(type(this_kpts0[0][0][0])))

            kpts0_new = this_kpts0
            kpts1_new = this_kpts1
            desc0_new = this_desc0.astype(np.float32)
            desc1_new = this_desc1.astype(np.float32)
            # desc0_new = norm_desc0.astype(np.float32)
            # desc1_new = norm_desc1.astype(np.float32)
            """
            matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "kpts0": kpts0_new,
                    "kpts1": kpts1_new,
                    "desc0": desc0_new,
                    "desc1": desc1_new,
                },
            )


            if self.model_type  == "onnx" and is_save_point:
                # print( " >>>>>> onnx > matches0 = {} <<<<<".format( matches0 ) )
                print(" >>>>>> onnx > matches0.shape = {} <<<<<".format(matches0.shape))
                prefix_matches_file = img0_path.split("/")[-1].split(".")[0] + "__" + \
                                      img1_path.split("/")[-1].split(".")[0]
                if not os.path.exists(self.onnx_matches_dir):
                    os.makedirs(self.onnx_matches_dir)
                np.savetxt('{0}/{1}.txt'.format(self.onnx_matches_dir, prefix_matches_file), matches0, fmt='%s',
                           delimiter='\t')

            elif self.model_type == "npu" and is_save_point:
                prefix_matches_file = img0_path.split("/")[-1].split(".")[0] + "__" + \
                                      img1_path.split("/")[-1].split(".")[0]
                matches_first = np.loadtxt('{0}{1}_first.txt'.format(self.npu_matches_dir, prefix_matches_file))
                matches_second = np.loadtxt('{0}{1}_second.txt'.format(self.npu_matches_dir, prefix_matches_file))

                # matches_first = np.loadtxt( "save_res/npu_matches/test_0223/img_165_70_0__img_165_70_1_first.txt")
                # matches_second = np.loadtxt( "save_res/npu_matches/test_0223/img_165_70_0__img_165_70_1_second.txt" )

                # matches_first = np.loadtxt("save_res/npu_matches/test_0223/psl_img_165_70_0__img_165_70_1_first.txt")
                # matches_second = np.loadtxt("save_res/npu_matches/test_0223/psl_img_165_70_0__img_165_70_1_second.txt")

                matches_list = []
                for i in range(matches_first.shape[0]):
                    if int(matches_first[i]) > norm_kpts_len or int(matches_second[i]) > norm_kpts_len:
                        continue
                    matches_list.append([int(matches_first[i]), int(matches_second[i])])
                matches0 = np.array(matches_list)


            else:
                pass

            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            if is_save_point:
                origin_kpts0, origin_kpts1 = self.post_process_this(kpts0_origin, kpts1_origin, scales0, scales1)
                img_name = img0_path.split("/")[-1].split(".")[0] + "__" + img1_path.split("/")[-1].split(".")[0]
                if not os.path.exists(
                        'save_res/result_points_{0}/{1}/'.format(self.model_type, self.image_cate)):
                    os.makedirs('save_res/result_points_{0}/{1}/'.format(self.model_type, self.image_cate))

                np.savetxt(
                    'save_res/result_points_{0}/{1}/{2}_topK_kpts0.txt'.format(self.model_type, self.image_cate,
                                                                                          img_name), origin_kpts0.squeeze(),
                    fmt='%s', delimiter='\t')
                np.savetxt(
                    'save_res/result_points_{0}/{1}/{2}_topK_kpts1.txt'.format(self.model_type, self.image_cate,
                                                                                          img_name), origin_kpts1.squeeze(),
                    fmt='%s', delimiter='\t')

                print(" >>>> desc0_new.shape = {} <<<<".format(desc0_new.shape))
                np.savetxt(
                    'save_res/result_points_{0}/{1}/{2}_topK_desc0.txt'.format(self.model_type, self.image_cate,
                                                                                          img_name), desc0_new.squeeze(),
                    fmt='%s', delimiter='\t')
                print(" >>>> desc1_new.shape = {} <<<<".format(desc1_new.shape))
                np.savetxt(
                    'save_res/result_points_{0}/{1}/{2}_topK_desc1.txt'.format(self.model_type, self.image_cate,
                                                                                          img_name), desc1_new.squeeze(),
                    fmt='%s', delimiter='\t')
            return m_kpts0, m_kpts1

    @staticmethod
    def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    @staticmethod
    def post_process(kpts0, kpts1, matches, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        print(" >>>>> kpts0.shape = {} <<<<<".format(kpts0.shape))
        print(" >>>>> kpts1.shape = {} <<<<<".format(kpts1.shape))
        print(" >>>>> matches.shape = {} <<<<<".format(matches.shape))
        print(" >>>>> type(matches) = {} <<<<<".format(type(matches)))
        kpts0_len = kpts0.shape[1]
        kpts1_len = kpts1.shape[1]
        res_match = []

        for match in matches:
            if (match[0] >= kpts0_len) or (match[1] >= kpts1_len):
                continue
            res_match.append(match)
        matches = np.array(res_match)

        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1

    @staticmethod
    def post_process_this(kpts0, kpts1, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        return kpts0, kpts1
