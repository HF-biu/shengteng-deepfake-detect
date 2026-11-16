import os
# import argparse
import numpy as np
import time
import yaml
# import model

from skimage import transform as trans
from utils.box_utils import np_decode, np_decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from PIL import Image

import cv2

from utils.ascend_runtime import StaticOMSession, ROOT



VISUALIZE = False
# parser = argparse.ArgumentParser(description='Process some paths.')
# parser.add_argument('--local_rank', type=int, default=-1)
# args = parser.parse_args()

import cv2
import numpy as np

from collections import OrderedDict

FAKE_THRESHOLD = 0.005  # Instances receive the configured threshold.

# import matplotlib.pyplot as plt
# import yaml 

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class FileStd():
    def __init__(self, state, id) -> None:
        self.state_list = []
        self.id = id
        self.update(state)

    def update(self, state):
        self.state_list.append(state)
    
    def collation(self):
        self.state_list.sort(key=lambda x:x[2])

def is_monochromatic(image_path, color_tone):
    np_image = np.array(image_path)
    
    red_channel = np_image[:,:,0]
    green_channel = np_image[:,:,1]
    blue_channel = np_image[:,:,2]
    
    if color_tone == 'red':
        condition = (red_channel > green_channel) & (red_channel > blue_channel)
    elif color_tone == 'green':
        condition = (green_channel > red_channel) & (green_channel > blue_channel)
    elif color_tone == 'blue':
        condition = (blue_channel > red_channel) & (blue_channel > green_channel)
    else:
        raise ValueError("Unsupported color tone. Choose from 'red', 'green', or 'blue'.")
    
    proportion = np.sum(condition) / condition.size
    return proportion > 0.9

def pre_select(image, threshold=15, percentage=0.85):
    image = image.astype(np.float32)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    max_diff = np.max(np.stack([
        np.abs(r - g),
        np.abs(r - b),
        np.abs(g - b)
    ], axis=-1), axis=2)
    bw_pixels = np.sum(max_diff < threshold)
    total_pixels = image.shape[0] * image.shape[1]

    state = bw_pixels / total_pixels >= percentage

    types = ['red', 'green', 'blue']
    for t in types:
        state = (state or is_monochromatic(image, t))
    return state

class ImageSegmentation:
    def __init__(self, original_image, ratio=1., face_box=[0,0,0,0],):
        # self.original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        self.original_image = Image.fromarray(original_image.astype(np.uint8))
        self.pre_sel = pre_select(original_image)
        width, height = self.original_image.size
        if width < 256 or height < 256:
            self.original_image = self.original_image.resize((256, 256), resample=Image.BICUBIC)
        self.sub_images = []
        self.coord = []
        self.masks = []
        self.split_image()
        self.cls = 0
        self.binary = 0
        self.combine_mask = None
        self.face_ratio = ratio
        self.face_box = face_box

    def split_image(self, patch_size=256, overlap=0):
        width, height = self.original_image.size
        stride = patch_size - overlap
        self.sub_images = []

        # Generate patches covering the entire image, overlapping at the edges
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x_end = min(x + patch_size, width)
                y_end = min(y + patch_size, height)
                x_start = max(0, x_end - patch_size)
                y_start = max(0, y_end - patch_size)
                box = (x_start, y_start, x_end, y_end)
                sub_image = self.original_image.crop(box)
                sub_image = Image.fromarray(np.uint8(sub_image))
                sub_image = np.asarray(sub_image)
                sub_image = (sub_image.astype(np.float32) / 255.0).transpose(2, 0, 1)
                # sub_image = torch.from_numpy(sub_image.astype(np.float32) / 255).permute(2, 0, 1)
                self.sub_images.append(sub_image)
                self.coord.append((x_start, y_start))

    def combine_masks(self, mask_size=(256, 256), overlap=0):
        width, height = self.original_image.size
        mask_combined = np.zeros((height, width), dtype=np.float32)
        count_mask = np.zeros((height, width), dtype=np.float32)
        stride = mask_size[0] - overlap

        for mask, (x, y) in zip(self.masks, [coord for coord in self.coord]):
            mask_array = np.array(mask)
            mask_combined[y:y+mask_size[1], x:x+mask_size[0]] += mask_array
            count_mask[y:y+mask_size[1], x:x+mask_size[0]] += 1

        # Normalize by the count_mask to handle overlaps
        count_mask[count_mask == 0] = 1  # Avoid division by zero
        mask_combined = (mask_combined / count_mask).astype(np.uint8)
        
        roi = mask_combined[self.face_box[1]:self.face_box[3], self.face_box[0]:self.face_box[2]]
        # self.cls = max(min(1.0, sum(sum((roi > 128))) / self.face_ratio), 0)
        # if not self.pre_sel: self.cls = 0.
        self.cls = max(min(1.0, np.sum(roi > 128) / max(self.face_ratio, 1.0)), 0)
        
        mask = Image.fromarray(mask_combined)
        overlay_mask = mask.convert('RGBA')
        r, g, b, a = overlay_mask.split()
        a = a.point(lambda i: i * 0.5)

        overlay_mask = Image.merge('RGBA', (r, g, b, a))
        self.combine_mask = self.original_image
        self.combine_mask.paste(overlay_mask, (0, 0), overlay_mask)
        return self.combine_mask

class ImageStd:
    def __init__(self, file, max_size=640) -> None:
        self.id = file.id
        self.frame_id = file.frame_id
        self.max_size = max_size
        self.type = file.type

        self.raw_image = file.content
        self.unified_image, init_image = self.unify_image(file.content)
        
        self.process_img, self.process_scale = self.init_face(init_image)
        self.faces = []
        self.boxes = []
        self.ps_roi = []
        self.face_ratio = 1.0

        self.state = 'Real'
        self.threshold = FAKE_THRESHOLD
        self.prob = np.array(0.0)  # 改为 numpy 数组
        self.out_face = None

        self.cam_img = None

    def init_face(self, img):
        # 修改后的 NumPy 实现
        img = img.astype(np.float32)
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        # 使用 NumPy 进行通道均值减法（需要广播）
        mean = np.array([[[104, 117, 123]]], dtype=np.float32)
        img = img - mean
        img = img.transpose(2, 0, 1)  # 通道重排
        
        return img, scale

    def unify_image(self, img):

        original_height, original_width = img.shape[:2]

        # 计算resize的比例，选择较小的比例以确保长宽都不超过最大阈值
        ratio = min(self.max_size / original_height, self.max_size / original_width)

        # # 新的高度和宽度
        new_height = int(original_height * ratio)
        new_width = int(original_width * ratio)

        # 使用cv2.resize按比例resize图像
        # if self.type == 'image': resized_img = cv2.resize(img, (self.max_size, self.max_size), cv2.INTER_CUBIC)
        # else: resized_img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
        resized_img = cv2.resize(img, (self.max_size, self.max_size), cv2.INTER_CUBIC)

        return img, resized_img

    def update_state(self, output, face):
        # print(output)
        if output > self.threshold: self.state = 'Fake'
        if self.out_face is None: self.out_face = face
        if self.prob < output:
            self.prob = output
            self.out_face = face
        # self.prob = max(self.prob, output)


class FaceStd():
    def __init__(self, belong_idx, face) -> None:
        self.belong_idx = belong_idx
        self.face = face
        self.res = None
    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)   

class Inference():
    def __init__(self, config) -> None:

        self.config = config
        self.PsNet = self.refine_clf = self.processers = None
        self.cam_model = None
        try:
            self.PsNet = StaticOMSession(config, 'pscc')
            self.refine_clf = StaticOMSession(config, 'deepfake')
            self.processers = Processer(config, None)
            self.retina_batch = self.processers.net.chunk_size
            self.ps_batch = self.PsNet.chunk_size
            self.deepfake_batch = self.refine_clf.chunk_size
            if config.get('ascend', {}).get('enable_cam', False):
                import torch
                from model.vision_transformer import RefineNet
                self.cam_model = RefineNet()
                weights = ROOT / config['deepfake_weights']
                self.cam_model.load_state_dict(torch.load(str(weights), map_location='cpu'), strict=True)
                self.cam_model.eval()
        except Exception:
            self.close()
            raise
        self.tic = time.time()
        self.results = []

    def close(self):
        for obj in (self.processers, self.refine_clf, self.PsNet):
            if obj is not None:
                obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def torch_normalize(self, img):
        """
        Normalize an image.
        """
        from torchvision import transforms as T
        img = T.ToTensor()(img)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def normalize(self, img):
        """
        使用 NumPy 实现的标准化流程
        输入：uint8 格式的 numpy 数组 (H, W, C)
        输出：float32 格式的 numpy 数组 (C, H, W)
        """
        img = np.array(img)
        # 转换像素值范围 [0,255] => [0.0,1.0]
        img = img.astype(np.float32) / 255.0
        
        # 转换维度顺序 HWC -> CHW
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        
        # 配置标准化参数
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(-1, 1, 1)
        
        # 执行标准化
        return (img - mean) / std
        
    def get_faces(self, data_list: ImageStd):
        # face detection
        batch_size = self.retina_batch
        
        print("Input Len:", len(data_list))
        debug_time = time.time()
        
        results = []
        
        batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

        for batch in batches:
            results.extend(self.processers.process_image(batch, batch_size=batch_size))
            
        print('face complete: {:.4f}'.format((time.time() - debug_time)))
        outputs = [None] * len(data_list)
        
        for i, i_results in enumerate(results):
            outputs[i] = i_results
            
        return outputs
    
    def get_grad_cam(self, inputs: ImageStd):
        if self.cam_model is None:
            for item in inputs:
                item.cam_img = np.asarray(item.out_face, dtype=np.uint8).copy()
            return
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # torch.set_grad_enabled(True) # required for grad cam

        target_layers = [self.cam_model.backbone.blocks[-1].norm1]
        res, x, faces = [], [], []
        targets = [ClassifierOutputTarget(1)]

        with EigenCAM(model=self.cam_model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform_vit, 
                    ) as cam:
            cam.batch_size = 1
            for i_input in inputs:
                out_face = i_input.out_face
                h, w = out_face.shape[0], out_face.shape[1]

                if i_input.state == 'Real':
                    grayscale_cam = np.zeros((h, w), dtype=np.float32)
                else:
                    ix = Image.fromarray(np.array(out_face, dtype=np.uint8))
                    ix = self.torch_normalize(ix).unsqueeze(0)

                    grayscale_cam = cam(input_tensor=ix,
                                        targets=targets,)[0, :]

                rgb_img = np.float32(out_face[:, :, ::-1]) / 255
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                i_input.cam_img = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


    def psinfer(self, seg_list):
        batch_size = self.ps_batch
        segs = []
        for seg in seg_list:
            segs.extend(seg.sub_images)

        results = []
        masks = []
        for outputs in self.PsNet.iter_batches(segs):
            if len(outputs) != 5:
                raise ValueError('PSCC OM must return four masks and classification logits')
            pred_mask1, pred_mask2, pred_mask3, pred_mask4, pred_logit = outputs
            # print(pred_mask1.shape)
            for pred_mask, logit in zip(pred_mask1, pred_logit):
                output_mask = Image.fromarray((pred_mask.squeeze() * 255).astype(np.uint8))
                masks.append(output_mask)
                logit = softmax(logit)
                results.append(logit[1])  # logit[1] is the probability of the positive class

        cnt_idx = 0
        for seg in seg_list:
            cnt_aim = len(seg.sub_images)
            while cnt_aim > 0 and cnt_idx < len(results):
                seg.masks.append(masks[cnt_idx])
                seg.binary = max(seg.binary, results[cnt_idx])
                cnt_idx += 1
                cnt_aim -= 1
            seg.combine_mask = seg.combine_masks()
        return

    def fake_infer(self, face_list):
        samples = [self.normalize(Image.fromarray(np.asarray(data.face, dtype=np.uint8)))
                   for data in face_list]
        results = []
        for outputs in self.refine_clf.iter_batches(samples):
            if not outputs:
                raise ValueError('Deepfake OM returned no outputs')
            predictions = outputs[0]
            if predictions.ndim == 2 and predictions.shape[1] == 2:
                logits = predictions - predictions.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                predictions = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
            elif predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions[:, 0]
            elif predictions.ndim != 1:
                raise ValueError('Deepfake output must have shape [N], [N,1] or [N,2]')
            if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
                raise ValueError('Deepfake model did not return valid probabilities')
            results.extend(predictions)
        return results


    def infer(self, inputs, batch_size=None, input_type='image'):
        if not inputs:
            return OrderedDict(), []
        self.results = np.zeros(len(inputs))

        # for ps detection...
        seg_list = []

        data_list = self.get_faces(inputs)

        face_list = []
        for i, i_data in enumerate(data_list):
            faces_i = i_data.faces
            ps_roi = i_data.ps_roi
            for k, face in enumerate(faces_i):
                face_list.append(FaceStd(i, face))
            seg_list.append(ImageSegmentation(ps_roi, i_data.face_ratio, i_data.boxes[0]))

        
        self.psinfer(seg_list)

        fake_outputs = self.fake_infer(face_list)

        for i, face_res in enumerate(face_list):
            # print(fake_outputs[i])
            face_res.res = fake_outputs[i]
            if np.all(face_res.face == 0): face_res.res = np.array(0.)
            data_list[face_res.belong_idx].update_state(face_res.res, face_res.face)

        self.get_grad_cam(data_list)

        res_dict = {}
        for i_data in data_list:
            key = i_data.id
            if key in res_dict:
                res_dict[key].update((i_data.state, i_data.prob, i_data.frame_id, i_data.out_face, i_data.cam_img))
            else:
                new_res = FileStd(state=(i_data.state, i_data.prob, i_data.frame_id, i_data.out_face, i_data.cam_img), id=i_data.id)
                res_dict[key] = new_res
                
            # print(i_data.state)
        
        for key in res_dict:
            res_dict[key].collation()
        res_dict = OrderedDict(sorted(res_dict.items()))

        return res_dict, seg_list

class Processer():
    def __init__(self, config, specific_device) -> None:
        

        self.threshold = config.get('deepfake_threshold', FAKE_THRESHOLD)
        self.ignore_threshold = config['face_detection']['ignore_threshold']
        self.top_k = config['face_detection']['top_k']
        self.nms_threshold = config['face_detection']['nms_threshold']
        self.keep_top_k = config['face_detection']['keep_top_k']
        self.face_threshold = config['face_detection']['face_threshold']
        self.limit_size = pow(config['face_detection']['limit_size'],2)
        self.net = StaticOMSession(config, 'retina')

    def close(self):
        self.net.close()

    @staticmethod
    def img_align_crop(img, landmark=None, outsize=(224, 224), scale=1.3):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        return np.float32(img)


    def get_face(self, raw, loc, conf, landms, scale):
        im_height, im_width, _ = raw.shape
        resize_h, resize_w = im_height / 640, im_width / 640

        priorbox = PriorBox(cfg_re50, image_size=(640, 640))
        priors = priorbox.forward()
        prior_data = np.asarray(priors.data)
        boxes = np_decode(loc, prior_data, cfg_re50['variance'])
        boxes = boxes * scale
        boxes[:, 0::2] *= resize_w
        boxes[:, 1::2] *= resize_h
        boxes = boxes

        scores = conf[:, 1]
        landms = np_decode_landm(landms, prior_data, cfg_re50['variance'])
        scale1 = np.array([640, 640, 640, 640, 640, 640, 640, 640, 640, 640])

        landms = landms * scale1
        landms[:, 0::2] *= resize_w
        landms[:, 1::2] *= resize_h
        landms = landms
        # ignore low scores
        inds = np.where(scores > self.ignore_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        face_list = []
        bbox_list = []
        for i, b in enumerate(dets):
            if b[4] < self.face_threshold:
                continue
            b = list(map(int, b))
            
            face = Processer.img_align_crop(raw, landmark=np.array([[b[5], b[6]],
                                                                    [b[7], b[8]],
                                                                    [b[9], b[10]],
                                                                    [b[11], b[12]],
                                                                    [b[13], b[14]]
                                                                    ]))
            bbox_list.append(b[:4])
            face_list.append(face)
        return face_list, bbox_list



    def face_detection(self, data_list: ImageStd):

        if not data_list:
            return []
        x = [np.asarray(item.process_img) for item in data_list]
        outputs = self.net.infer_batch(x)
        if len(outputs) != 3:
            raise ValueError('RetinaFace OM must return loc, conf, landms')
        loc, conf, landms = outputs

        for i in range(len(data_list)):
            faces, bboxes = self.get_face(data_list[i].unified_image, loc[i], conf[i], landms[i], data_list[i].process_scale)

            # print(bboxes)
            qualified_faces = []
            
            limit_size = -1
            for face, box in zip(faces, bboxes):
                width, height = box[2] - box[0], box[3] - box[1]
                limit_size = max(limit_size, width*height)
            limit_size = min(limit_size, self.limit_size)
            
            max_face, max_size, max_box = None, -1, None
            for face, box in zip(faces, bboxes):
                width, height = box[2] - box[0], box[3] - box[1]
                if width * height < limit_size:
                    continue
                area = width * height
                if area > max_size:
                    max_size = area
                    max_face = face
                    # max_box = box
                    ori_h, ori_w, _ = data_list[i].unified_image.shape
                    center_w, center_h = int((box[2] + box[0]) // 2), int((box[3] + box[1]) // 2)
                    w_stx, h_stx = max(0, center_w - 128), max(0, center_h - 128)
                    w_end, h_end = min(ori_w, w_stx + 256), min(ori_h, h_stx + 256)
                    # print(f'({w_stx}, {w_end}), ({h_stx}, {h_end}) < ({ori_w}, {ori_h}), {box}, ({center_w}, {center_h}) = {area}')
                    try:
                        ps_roi = data_list[i].unified_image[h_stx:h_end, w_stx:w_end]
                        ps_roi = cv2.resize(ps_roi, (256, 256))    
                    except Exception as e:
                        print(f'{data_list[i].unified_image.shape}, ({w_stx}, {w_end}), ({h_stx}, {h_end}), {e}')
                        ps_roi = np.zeros((256, 256, 3)) * 256

                    max_box = [box[0] - w_stx, box[1] - h_stx, box[2] - w_stx, box[3] - h_stx]

            if max_face is not None: qualified_faces.append((max_face, max_box, ps_roi))
            
            if len(qualified_faces) == 0 or max_face is None:
                # FIXME: bad choice
                max_face = np.zeros((224, 224, 3))
                max_box = [0,0,0,0]
                ps_roi = np.zeros((256, 256, 3))
                print("WARNING: Frame {} has no qualified face!".format(data_list[i].frame_id))
   
            if max_size < 120 * 120:
                max_face = np.zeros((224, 224, 3))

            data_list[i].faces = [max_face]
            data_list[i].boxes = [max_box]
            data_list[i].ps_roi = ps_roi
            data_list[i].face_ratio = max_size
            # break

        return data_list

    def process_image(self, file_list, batch_size=16, input_type='image'):

        img_list = []
        for i in range(min(batch_size, len(file_list))):
            
            # FileStd
            file = file_list[i]
 
            if file.content is None:
                raise ValueError('Input image content is None')
            item = ImageStd(file)
            item.threshold = self.threshold
            img_list.append(item)

        file_list = self.face_detection(img_list)
        
        return file_list

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
     

import glob
if __name__ == '__main__':
    config = {}
    

    test = Inference(config)
    directory = '/data/huangwen/worklab/AdaptiveCross/datasets/ps'
    image_paths = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.png'))
    print(image_paths)
    # test.process_image(image_paths=image_files)
    # test.infer(image_paths)

