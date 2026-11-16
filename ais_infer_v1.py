import os
# import argparse
import numpy as np
import time

# import model

from skimage import transform as trans
from utils.box_utils import np_decode, np_decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from PIL import Image

import cv2

from ais_bench.infer.interface import InferSession, MultiDeviceSession

device = 'cuda'

VISUALIZE = False
# parser = argparse.ArgumentParser(description='Process some paths.')
# parser.add_argument('--local_rank', type=int, default=-1)
# args = parser.parse_args()

import cv2
import numpy as np

from collections import OrderedDict

from model.vision_transformer import RefineNet

import torch
from torchvision import transforms as T

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
        sorted(self.state_list, key=lambda x:x[2])

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
        mask_combined = np.zeros((height, width), dtype=np.uint8)
        count_mask = np.zeros((height, width), dtype=np.uint8)
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
        self.cls = max(min(1.0, np.sum(roi > 128) / self.face_ratio), 0)
        
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
        self.threshold = 0.005
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

        self.retina_batch = config['batch_size']['retina']
        self.ps_batch = config['batch_size']['pscc']
        self.deepfake_batch = config['batch_size']['deepfake']

        self.num_gpus = 1
        
        self.prepare_net()

        self.cam_model = RefineNet()
        self.cam_model.load_state_dict(torch.load(config['deepfake_weights'], 'cpu'), strict=False)
        self.cam_model.eval()

        print("Init model...  Done. \n GPU counts:", self.num_gpus)
        
        self.processers = Processer(config, None)

        self.tic = time.time()

        self.results = []

    def prepare_net(self):
        self.PsNet = InferSession(device_id=0, model_path="./pscc.om")
        self.refine_clf = MultiDeviceSession("./refine_128.om")

    def torch_normalize(self, img):
        """
        Normalize an image.
        """
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
        # torch.set_grad_enabled(True) # required for grad cam

        target_layers = [self.cam_model.backbone.blocks[-1].norm1]
        res, x, faces = [], [], []
        targets = [ClassifierOutputTarget(1)]
        for i_input in inputs:
            res.append(i_input.prob)
            ix = np.array(i_input.out_face, dtype=np.uint8)
            ix = Image.fromarray(ix)
            ix = self.torch_normalize(ix)
            x.append(ix)
            faces.append(i_input.out_face)
            

        with EigenCAM(model=self.cam_model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform_vit, 
                    ) as cam:
            cam.batch_size = 1
            for i in range(len(x)):
                ix = x[i].unsqueeze(0)
                # print(ix.shape)
                grayscale_cam = cam(input_tensor=ix,
                                    targets=targets,)
                grayscale_cam = grayscale_cam[0, :]
                if res[i] < 0.7: grayscale_cam = np.zeros_like(grayscale_cam)
                rgb_img = faces[i][:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                inputs[i].cam_img = cam_image

    def psinfer(self, seg_list):
        batch_size = self.ps_batch
        segs = []
        for seg in seg_list:
            segs.extend(seg.sub_images)

        batches = [segs[i:i + batch_size] for i in range(0, len(segs), batch_size)]
        # 计算需要补充的数组数量
        num_needed = batch_size - len(batches[-1])

        if num_needed > 0:
            array_to_copy = np.zeros((3, 256, 256)) 
            for _ in range(num_needed):
                batches[-1].append(array_to_copy.copy())

        results = []
        masks = []
        for i, batch in enumerate(batches):
            x = [np.stack(batch).astype(np.float32)]
            [pred_mask1, pred_mask2, pred_mask3, pred_mask4, pred_logit] = self.PsNet.infer(feeds=x, mode='static')
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
        batch_size = self.deepfake_batch
        device_batches = [[] for _ in range(self.num_gpus)]
        device_ids = [[] for _ in range(self.num_gpus)]


        for i, data in enumerate(face_list):
            ix = np.array(data.face, dtype=np.uint8)
            ix = Image.fromarray(ix)
            ix = self.normalize(ix)
            device_batches[i % self.num_gpus].append(ix)
            device_ids[i % self.num_gpus].append(i)
        


        outputs = [None] * len(face_list)

        device_feeds = [[] for _ in range(self.num_gpus)]
        device_feed_ids = [[] for _ in range(self.num_gpus)]
        for i, (dbatch, did) in enumerate(zip(device_batches, device_ids)):
            device_feeds[i] = [dbatch[i:i + batch_size] for i in range(0, len(dbatch), batch_size)]
            device_feed_ids[i] = [did[i:i + batch_size] for i in range(0, len(did), batch_size)]

        for t in range(self.num_gpus):

            num_needed = batch_size - len(device_feeds[t][-1])
            # 如果需要补充，则复制现有的数组，直到达到64个
            if num_needed > 0:
                # 可以选择一个特定的数组来复制，这里选择第一个数组（假设列表不为空）
                # 如果列表为空，你可能需要定义一个默认的数组
                array_to_copy = np.zeros((224, 224, 3))  # 假设默认数组形状为 (1,)
                array_to_copy = np.array(array_to_copy, dtype=np.uint8)
                array_to_copy = Image.fromarray(array_to_copy)
                array_to_copy = self.normalize(array_to_copy)
                # 复制数组 num_needed 次
                for _ in range(num_needed):
                    device_feeds[t][-1].append(array_to_copy.copy())
                    device_feed_ids[t][-1].append(-1)

        max_len = max([len(dfeeds) for dfeeds in device_feeds])
        for i in range(max_len):
            valid_feeds = {}
            valid_ids = {}
            for t in range(self.num_gpus):
                if i+1 < max_len or (i+1 == max_len and len(device_feeds[t]) == max_len):
                    valid_feeds[t+1] = [[np.stack(device_feeds[t][i]).astype(np.float32)]]
                    # print(np.stack(device_feeds[t][i]).astype(np.float32))
                    valid_ids[t+1] = device_feed_ids[t][i]
            i_output = self.refine_clf.infer(valid_feeds, mode='static')

            for gpu in valid_feeds:
                # print(len(i_output[gpu][0]), ":", i_output[gpu])
                # print()
                for i, res in enumerate(i_output[gpu][0][0]):
                    if valid_ids[gpu][i] == -1: continue
                    # print(i, res)
                    if res.shape[0] == 2:
                        max_vals = np.max(res)
                        exp_pred = np.exp(res - max_vals)
                        softmax_output = exp_pred / exp_pred.sum()
                        res = softmax_output[1]
                    outputs[valid_ids[gpu][i]] = res
                # print
        # print(outputs)
        return outputs


    def infer(self, inputs, batch_size=None, input_type='image'):
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
            cv2.rectangle(ps_roi, (i_data.boxes[0][0], i_data.boxes[0][1]), (i_data.boxes[0][2], i_data.boxes[0][3]), (255, 0, 0), 5)
            cv2.imwrite('./mask/'+str(i)+'.jpg', ps_roi)
        
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
        

        self.net = InferSession(device_id=0, model_path="./FaceDetector.om")
        self.ignore_threshold = config['face_detection']['ignore_threshold']
        self.top_k = config['face_detection']['top_k']
        self.nms_threshold = config['face_detection']['nms_threshold']
        self.keep_top_k = config['face_detection']['keep_top_k']
        self.face_threshold = config['face_detection']['face_threshold']
        self.limit_size = pow(config['face_detection']['limit_size'],2)

        self.num_gpus = 2
        # self.device = specific_device

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

        # print(len(img_list))
        
        # x = np.asarray([np.asarray(data_list[0].process_img), np.asarray(data_list[1].process_img)]).astype(np.float64)
        x = [np.asarray(i_data.process_img) for i_data in data_list]
        num_needed = 64 - len(x)

        if num_needed > 0:
            array_to_copy = np.zeros((3, 640, 640))
            for _ in range(num_needed):
                x.append(array_to_copy.copy())
        x = [np.stack(x, axis=0).astype(np.float32)]
        # print(self.net.get_inputs()[0].shape)
        # x = [np.full(self.net.get_inputs()[0].shape, 1).astype(np.float32)]

        loc, conf, landms = self.net.infer(feeds=x, mode="static")

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
            cv2.imwrite('debug/'+str(i)+'.png', ps_roi)
        return data_list

    def process_image(self, file_list, batch_size=16, input_type='image'):

        img_list = []
        for i in range(min(batch_size, len(file_list))):
            
            # FileStd
            file = file_list[i]
 
            if file.content is None:
                print('Got NoneType content, continue...')
                continue
            img_list.append(ImageStd(file))

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

