import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time

import model

from skimage import transform as trans
from timm.models import create_model
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from torchvision import transforms as T
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from PIL import Image

import cv2
from torchvision import transforms
from models.retinaface import RetinaFace

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from collections import OrderedDict

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args
import torchvision.utils as tv_utils

device = 'cuda'

VISUALIZE = False
# parser = argparse.ArgumentParser(description='Process some paths.')
# parser.add_argument('--local_rank', type=int, default=-1)
# args = parser.parse_args()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml 
from model import DETECTOR

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from model.refine_cfl import RefineNet

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_fourier_spectrum(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # 计算幅度谱（频谱）
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Fourier Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')

    plt.savefig("report.png")

class FileStd():
    def __init__(self, state, id) -> None:
        self.state_list = []
        self.id = id
        self.update(state)

    def update(self, state):
        self.state_list.append(state)
    
    def collation(self):
        sorted(self.state_list, key=lambda x:x[2])

def pre_select(image, threshold=15, percentage=0.85):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    max_diff = np.max(np.stack([
        np.abs(r - g),
        np.abs(r - b),
        np.abs(g - b)
    ], axis=-1), axis=2)
    bw_pixels = np.sum(max_diff < threshold)
    total_pixels = image.shape[0] * image.shape[1]
    return bw_pixels / total_pixels >= percentage

class ImageSegmentation:
    def __init__(self, original_image, ratio=1., face_box=[0,0,0,0],):
        # self.original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        self.pre_sel = pre_select(original_image)
        self.original_image = Image.fromarray(original_image.astype(np.uint8))
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
                sub_image = torch.from_numpy(sub_image.astype(np.float32) / 255).permute(2, 0, 1)
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
        
        # roi = mask_combined[self.face_box[1]:self.face_box[3], self.face_box[0]:self.face_box[2]]
        self.cls = max(min(1.0, np.sum(mask_combined > 128) / self.face_ratio), 0.)
        if not self.pre_sel: self.cls = 0.
        
        self.cls = max(0, self.cls)

        mask = Image.fromarray(mask_combined)
        overlay_mask = mask.convert('RGBA')
        r, g, b, a = overlay_mask.split()
        a = a.point(lambda i: i * 0.5)

        overlay_mask = Image.merge('RGBA', (r, g, b, a))
        self.combine_mask = self.original_image
        self.combine_mask.paste(overlay_mask, (0, 0), overlay_mask)
        return self.combine_mask

class ImageStd():

    def __init__(self, file, max_size = 640) -> None:
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
        self.face_ratio = 1.

        self.state = 'Real'
        self.threshold = 0.5
        self.prob = torch.tensor(0.)
        self.out_face = None

        self.cam_img = None

    def init_face(self, img):

        img = np.float32(img)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

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
    
        

class Inference():
    def __init__(self, config) -> None:

        # with open('./objectformer_bs24_lr2.5e-4_imd20.yaml', 'r') as f:
        #     cfg = yaml.safe_load(f)
        
        # self.model3 = ObjectFormer(cfg['MODEL'])
        # self.model3.load_state_dict(torch.load('weights/casiav2_and_imd20_ep40_lr2.5e-4.pth', 'cpu')['model_state'])
        # self.model3 = self.model3.to(device)
        # self.model3.eval()

        self.retina_batch = config['batch_size']['retina']
        self.ps_batch = config['batch_size']['pscc']
        self.deepfake_batch = config['batch_size']['deepfake']

        self.num_gpus = torch.cuda.device_count()
        self.model_tuple = [self.prepare_single_model(torch.device(f'cuda:{i}'), config['deepfake_weights']) for i in range(self.num_gpus)]
        # self.model = self.model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     self.model = torch.nn.DataParallel(self.model)
        # self.model.eval()

        self.cam_model = RefineNet()
        self.cam_model.load_state_dict(torch.load(config['deepfake_weights'], 'cpu'), strict=True)
        self.cam_model.eval()

        self.prepare_ps_net()

        print("Init model...  Done. \n GPU counts:", self.num_gpus)
        # with open('./spsl.yaml', 'r') as f:
        #     config = yaml.safe_load(f)
        # model_class = DETECTOR['spsl']
        # self.model2 = model_class(config)
        # self.model2 = self.model2.to(device)
        # self.model2.eval()

        self.processers = [Processer(config, torch.device(f'cuda:{i}')) for i in range(self.num_gpus)]

        self.tic = time.time()

        torch.set_grad_enabled(False)
        torch.set_printoptions(sci_mode=False)

        self.results = []

    def prepare_ps_net(self):
        # define backbone
        FENet_name = 'HRNet'
        FENet_cfg = get_hrnet_cfg()
        self.FENet = get_seg_model(FENet_cfg)

        args = get_pscc_args()
        # define localization head
        SegNet_name = 'NLCDetection'
        self.SegNet = NLCDetection(args)

        # define detection head
        ClsNet_name = 'DetectionHead'
        self.ClsNet = DetectionHead(args)

        self.FENet = self.FENet.to(device)
        self.SegNet = self.SegNet.to(device)
        self.ClsNet = self.ClsNet.to(device)

        self.FENet = torch.nn.DataParallel(self.FENet)
        self.SegNet = torch.nn.DataParallel(self.SegNet)
        self.ClsNet = torch.nn.DataParallel(self.ClsNet)

        # FENet_checkpoint = torch.load("./ckpt/HRNet_checkpoint/HRNet_30.pth")
        # FENet_checkpoint = torch.load("./ckpt/HRNet_checkpoint/HRNet_41_test.pth")
        # FENet_checkpoint = torch.load("./good_ckpt/20240705-2/test_HRNet.pth")
        FENet_checkpoint = torch.load("./good_ckpt/20240716/HRNet_checkpoint/HRNet_21_test.pth")
        # self.FENet.load_state_dict(FENet_checkpoint)
        self.FENet.load_state_dict(FENet_checkpoint['model'])

        # SegNet_checkpoint = torch.load("./ckpt/NLCDetection_checkpoint/NLCDetection_30.pth")
        # SegNet_checkpoint = torch.load("./ckpt/NLCDetection_checkpoint/NLCDetection_41_test.pth")
        # SegNet_checkpoint = torch.load("./good_ckpt/20240705-2/test_NLCDetection.pth")
        SegNet_checkpoint = torch.load("./good_ckpt/20240716/NLCDetection_checkpoint/NLCDetection_21_test.pth")
        # self.SegNet.load_state_dict(SegNet_checkpoint)
        self.SegNet.load_state_dict(SegNet_checkpoint['model'])

        # ClsNet_checkpoint = torch.load("./ckpt/DetectionHead_checkpoint/DetectionHead_30.pth")
        # ClsNet_checkpoint = torch.load("./ckpt/DetectionHead_checkpoint/DetectionHead_41_test.pth")
        # ClsNet_checkpoint = torch.load("./good_ckpt/20240705-2/test_DetectionHead.pth")
        ClsNet_checkpoint = torch.load("./good_ckpt/20240716/DetectionHead_checkpoint/DetectionHead_21_test.pth")
        # self.ClsNet.load_state_dict(ClsNet_checkpoint)
        self.ClsNet.load_state_dict(ClsNet_checkpoint['model'])

        self.FENet.eval()
        self.SegNet.eval()
        self.ClsNet.eval()


    def prepare_single_model(self, device, weights):
        model = RefineNet()
        # model = create_model(
        #     'resnet34',
        #     num_classes=2,
        # )
        model.load_state_dict(torch.load(weights, 'cpu'), strict=True)
        model.to(device)
        model.eval()
        
        return model

    def normalize(self, img):
        """
        Normalize an image.
        """
        img = T.ToTensor()(img)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)
    
    def get_grad_cam(self, inputs: ImageStd):
        torch.set_grad_enabled(True) # required for grad cam

        target_layers = [self.cam_model.backbone.blocks[-1].norm1]
        res, x, faces = [], [], []
        targets = [ClassifierOutputTarget(1)]
        for i_input in inputs:
            res.append(i_input.prob)
            ix = np.array(i_input.out_face, dtype=np.uint8)
            ix = Image.fromarray(ix)
            ix = self.normalize(ix)
            x.append(ix)
            faces.append(i_input.out_face)
            

        with EigenCAM(model=self.cam_model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform_vit, 
                    ) as cam:
            cam.batch_size = 1
            for i in range(len(x)):
                ix = x[i].unsqueeze(0).requires_grad_(True)
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

    def device_infer(self, batch_data, device_id):
        specific_device = torch.device(f'cuda:{device_id % self.num_gpus}')
        inputs = []
        for i_data in batch_data:
            ix = np.array(i_data.face, dtype=np.uint8)
            ix = Image.fromarray(ix)
            ix = self.normalize(ix)
            inputs.append(ix)
        with torch.no_grad():
            outputs = self.model_tuple[device_id](torch.stack(inputs).to(specific_device))
            outputs = outputs.cpu() # Move outputs back to CPU
            if len(outputs.shape) == 2:
                softmax = torch.nn.Softmax(dim=-1)
                outputs = softmax(outputs)[:, 1].contiguous()
            
            for i, output in enumerate(outputs):
                print(output)
                batch_data[i].res = output
        torch.cuda.empty_cache()
        return batch_data

    def process_batch(self, data_list: FaceStd):
        # for deepfake detection
        batch_size = self.deepfake_batch

        device_batches = [[] for _ in range(self.num_gpus)]
        device_ids = [[] for _ in range(self.num_gpus)]
        for i, data in enumerate(data_list):
            # device = torch.device(f'cuda:{i % self.num_gpus}')
            device_batches[i % self.num_gpus].append(data)
            device_ids[i % self.num_gpus].append(i)


        futures = [[] for _ in range(self.num_gpus)]
        for i, dbatch in enumerate(device_batches):
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Future 对象列表
                if not dbatch:  # Skip if no data for this GPU
                    continue
                batches = [dbatch[i:i + batch_size] for i in range(0, len(dbatch), batch_size)]
                for batch in batches:
                    futures[i].append(executor.submit(self.device_infer, batch, i))
        
        results = [[] for _ in range(self.num_gpus)]
        for i, i_future in enumerate(futures):
            if not i_future:  # Skip if no data for this GPU
                continue
            for j, future in enumerate(i_future):
                result = future.result()    # 阻塞直到结果返回
                results[i].extend(result)

        outputs = [None] * len(data_list)
        for i, i_results in enumerate(results):
            for j in range(len(i_results)):
                outputs[device_ids[i][j]] = i_results[j]

        return outputs
    
    def device_process(self, data, device_id, batch_size):
        with torch.no_grad():
            data = self.processers[device_id].process_image(data, batch_size=batch_size)
        torch.cuda.empty_cache()
        return data
        
    def get_faces(self, data_list: ImageStd):
        # face detection
        batch_size = self.retina_batch
        device_batches = [[] for _ in range(self.num_gpus)]
        device_data = [[] for _ in range(self.num_gpus)]
        for i, data in enumerate(data_list):
            # device = torch.device(f'cuda:{i % self.num_gpus}')
            device_batches[i % self.num_gpus].append(data)
            device_data[i % self.num_gpus].append(i)
        
        print("Input Len:", len(data_list))
        debug_time = time.time()
        # Future 对象列表
        futures = [[] for _ in range(self.num_gpus)]
        results = [[] for _ in range(self.num_gpus)]
        for i, dbatch in enumerate(device_batches):
            if not dbatch:  # Skip if no data for this GPU
                continue
            with ThreadPoolExecutor(max_workers=2) as executor:
                batches = [dbatch[i:i + batch_size] for i in range(0, len(dbatch), batch_size)]
                for batch in batches:
                    # thread = threading.Thread(target=lambda: results[i].extend(self.device_process(batch, i, batch_size)))
                    # thread.start()
                    # futures[i].append(thread)
                    # print(f"Device {i}: {len(dbatch)}")
                    
                    #     print(f"sub Device {i}: {len(batch)}")
                    futures[i].append(executor.submit(self.device_process, batch, i, batch_size))
            
        # for i, i_future in enumerate(futures):
        #     if not i_future:  # Skip if no data for this GPU
        #         continue
        #     for future in i_future:
        #         future.join()

        for i, i_future in enumerate(futures):
            if not i_future:  # Skip if no data for this GPU
                continue
            for future in i_future:
                result = future.result()
                results[i].extend(result)
                    
        print('face complete: {:.4f}'.format((time.time() - debug_time)))
        outputs = [None] * len(data_list)
        temp = 0
        for i, i_results in enumerate(results):
            for j in range(len(i_results)):
                temp += 1
                outputs[device_data[i][j]] = i_results[j]
        # print("Out len", temp)

        return outputs
    
    def psinfer(self, seg_list):
        # for ps artifacts detection
        batch_size = self.ps_batch
        segs = []
        for seg in seg_list:
            segs.extend(seg.sub_images)

        batches = [segs[i:i + batch_size] for i in range(0, len(segs), batch_size)]
        results = []
        masks = []
        for i, batch in enumerate(batches):
            x = torch.stack(batch).cuda()
            with torch.no_grad():
                feat = self.FENet(x)
                pred_logit = self.ClsNet(feat)
                [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = self.SegNet(feat)
            for pred_mask, logit in zip(pred_mask1, pred_logit):
                # print(pred_mask.shape, logit.shape, pred_logit.shape, logit, pred_logit)
                output_mask = Image.fromarray((pred_mask.squeeze().detach().cpu().numpy()*255).astype(np.uint8))
                masks.append(output_mask)
                softmax = torch.nn.Softmax(0)
                logit = softmax(logit)
                # _, binary_cls = torch.max(logit, 0)
                # if binary_cls.item() == 0:
                #     output_txt = 1
                # else:
                #     output_txt = 0
                results.append(logit[1])
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
    
    def infer(self, inputs, batch_size=None, input_type='image'):
        self.results = np.zeros(len(inputs))

        # for ps detection...
        seg_list = []
        # for i in range(len(inputs)):
        #     seg_list.append(ImageSegmentation(inputs[i].content))

        # self.psinfer(seg_list)

        data_list = self.get_faces(inputs)

        face_list = []
        for i, i_data in enumerate(data_list):
            faces_i = i_data.faces
            ps_roi = i_data.ps_roi
            for k, face in enumerate(faces_i):
                face_list.append(FaceStd(i, face))
            seg_list.append(ImageSegmentation(ps_roi, i_data.face_ratio, i_data.boxes[0]))
            # cv2.rectangle(ps_roi, (i_data.boxes[0][0], i_data.boxes[0][1]), (i_data.boxes[0][2], i_data.boxes[0][3]), (255, 0, 0), 5)
            cv2.imwrite('./temp/'+str(i)+'.jpg', faces_i[0])

        self.psinfer(seg_list)

        face_list = self.process_batch(face_list)
        for face_res in face_list:
            if np.all(face_res.face == 0): face_res.res = torch.tensor(0.)
            data_list[face_res.belong_idx].update_state(face_res.res, face_res.face)

        self.get_grad_cam(data_list)

        res_dict = {}
        for i_data in data_list:
            key = i_data.id
            if key in res_dict:
                res_dict[key].update((i_data.state, i_data.prob, i_data.frame_id, i_data.cam_img))
            else:
                new_res = FileStd(state=(i_data.state, i_data.prob, i_data.frame_id, i_data.cam_img), id=i_data.id)
                res_dict[key] = new_res
                
            # print(i_data.state)
        
        for key in res_dict:
            res_dict[key].collation()
        res_dict = OrderedDict(sorted(res_dict.items()))

        
        ''' 
        # cnt = 0
        # for i in inputs:
        #     face_i = i.faces
            
        #     for k, face in enumerate(face_i):
        #         cv2.imwrite('/data/huangwen/worklab/AgentViT/debug/'+str(cnt)+'.jpg', face)
        #         cnt += 1

        # all_faces = []
        # all_ids = []
        # all_bboxes = []
        # all_frame = {}
        # for i in range((len(inputs)-1) // batch_size + 1):
        #     i_img = inputs[i*batch_size : min(len(inputs), (i+1)*batch_size)]
        #     # print(len(i_path), len(img_paths), i*batch_size)
        #     x, ids, bboxes, frame_map = self.processer.process_image(i_img, batch_size)
        
        #     all_faces.extend(x)
        #     all_ids.extend(ids)
        #     all_bboxes.extend(bboxes)
        #     all_frame.update(frame_map)

        # for key in all_frame:
        #     frame_path = all_frame[key]
        #     all_frame[key] = cv2.imread(frame_path)

        
        # all_res = []
        # all_res2 = []
        # all_res3 =  []
        # for i in range((len(all_faces)-1) // batch_size + 1):
        #     ix = all_faces[i*batch_size : min(len(all_faces), (i+1)*batch_size)]
        #     # print(len(ix))
        #     for i, i_ix in enumerate(ix):
        #         ix[i] = np.array(i_ix, dtype=np.uint8)
        #         ix[i] = self.normalize(ix[i])
        #     ix = torch.stack(ix, dim=0)
        #     ix = ix.to(device)
        #     ix1 = self.model(ix)
        #     ix2 = self.model2(ix)
        #     ix3 = self.model3(ix)
        #     # print('net forward time: {:.4f} avg {:.4f}'.format((time.time() - self.tic), (time.time() - self.tic) / len(img_paths)))
        #     all_res.extend(ix1.cpu().numpy())
        #     all_res2.extend(ix2.cpu().numpy())
        #     all_res3.extend(ix3[0].cpu().numpy())
        '''
        
        # if input_type == 'video':
        #     '''
        #     max_res_dict = {}
        #     # 遍历all_res和all_ids
        #     original_height, original_width = all_frame[0].shape[:2]

        #     # 计算resize的比例，选择较小的比例以确保长宽都不超过最大阈值
        #     ratio = min(640 / original_height, 640 / original_width)
        #     for res, vid, bbox in zip(all_res, all_ids, all_bboxes):
        #         if vid not in max_res_dict:
        #             # 如果id不在字典中，添加id并将对应值设为当前res
        #             max_res_dict[vid] = res
        #         else:
        #             # 如果id已经在字典中，更新对应值为当前res和已有值的最大值
        #             max_res_dict[vid] = max(max_res_dict[vid], res)

        #         if VISUALIZE:
        #             bbox = [int(b / ratio) for b in bbox]
        #             if res > 0.3:
        #                 cv2.rectangle(all_frame[vid], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        #                 text = 'Fake Face'
        #             else:
        #                 text = 'Real Face'
        #                 cv2.rectangle(all_frame[vid], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #             cx = bbox[0]
        #             cy = bbox[1] + 12
        #             cv2.putText(all_frame[vid], text, (cx, cy),
        #                                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #     # 遍历字典，生成results列表
        #     unique_ids = list(set(all_ids))
        #     if VISUALIZE:
        #         video = cv2.VideoWriter('outmasike.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (all_frame[0].shape[1], all_frame[0].shape[0]))
        #         for i in range(len(unique_ids)):
        #             # cv2.imwrite("video/"+str(i)+'.png', all_frame[i])
        #             video.write(all_frame[i])
        #         video.release()

        #     print(max(unique_ids), len(max_res_dict))
        #     self.results = [max_res_dict[vid] for vid in range(max(unique_ids))]
        #     for i, res in enumerate(all_res):
        #         print(i, res)
        #     return self.results
        #     '''
        #     # for i, key in enumerate(res_dict):
        #     #     for j, res in enumerate(res_dict[key].state_list):
        #     #         print(j, ":", res[0], "{:.2f}".format(res[1]))
        #     # print()
        #     pass
        # else:
        # print("检测结果：", end=' ')
        # for i, key in enumerate(res_dict):
        #     print(i, ":", res_dict[key].state_list[0][0], "{:.2f}".format(res_dict[key].state_list[0][1].item()), 'PS检测：', "{:.2f}".format(seg_list[i].cls))
        # print()
        return res_dict, seg_list

def phase_without_amplitude(self, img):
    # Convert to grayscale
    gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
    # Compute the DFT of the input signal
    X = torch.fft.fftn(gray_img)
    # Extract the phase information from the DFT
    phase_spectrum = torch.angle(X)
    # Create a new complex spectrum with the phase information and zero magnitude
    reconstructed_X = torch.exp(1j * phase_spectrum)
    # Use the IDFT to obtain the reconstructed signal
    reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
    return reconstructed_x


class Processer():
    def __init__(self, config, specific_device) -> None:
        self.net = RetinaFace(cfg=cfg_re50, phase = 'test')
        self.net = load_model(self.net, 'weights/Resnet50_Final.pth', True)
        self.net = self.net.to(specific_device)
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     self.net = torch.nn.DataParallel(self.net)
        self.net.eval()   

        self.pre_net = create_model('resnet50', num_classes=2)
        state_dict = torch.load('weights/resnet_test_model.pth', 'cpu')

        self.pre_net.load_state_dict(state_dict)
        self.pre_net.to(specific_device)
        self.pre_net.eval()

        self.ignore_threshold = config['face_detection']['ignore_threshold']
        self.top_k = config['face_detection']['top_k']
        self.nms_threshold = config['face_detection']['nms_threshold']
        self.keep_top_k = config['face_detection']['keep_top_k']
        self.face_threshold = config['face_detection']['face_threshold']
        self.limit_size = pow(config['face_detection']['limit_size'],2)

        self.num_gpus = torch.cuda.device_count()
        self.device = specific_device

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

    def device_infer(self, data, device_id):
        with torch.no_grad():
            output = self.model_tuple[device_id](torch.stack(data))
            return output.cpu() # Move outputs back to CPU

    def get_face(self, raw, loc, conf, landms, scale):
        im_height, im_width, _ = raw.shape
        # print(raw.shape)
        resize_h, resize_w = im_height / 640, im_width / 640

        priorbox = PriorBox(cfg_re50, image_size=(640, 640))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
        boxes = boxes * scale.to(boxes.device)
        boxes[:, 0::2] *= resize_w
        boxes[:, 1::2] *= resize_h
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
        # scale1 = torch.Tensor([raw.shape[1], raw.shape[0], raw.shape[1], raw.shape[0],
        #                     raw.shape[1], raw.shape[0], raw.shape[1], raw.shape[0],
        #                     raw.shape[1], raw.shape[0]])
        scale1 = torch.Tensor([640, 640, 640, 640, 640, 640, 640, 640, 640, 640])

        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms[:, 0::2] *= resize_w
        landms[:, 1::2] *= resize_h
        landms = landms.cpu().numpy()
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


        x = torch.stack([i_data.process_img for i_data in data_list], dim=0).to(self.device)
        with torch.no_grad():
            loc, conf, landms = self.net(x)
        with torch.no_grad():
            pre_res = self.pre_net(x)

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

            _, predicted = torch.max(pre_res[i:i+1], 1)
            print(predicted)
            # if predicted == 1:
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
                        ps_roi = cv2.cvtColor(ps_roi, cv2.COLOR_BGR2RGB)
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
                
            # else:
                # cv2.imwrite('temp/' + str(frame_num[i])+'.png', faces[0])
            if max_size < 120 * 120:
                max_face = np.zeros((224, 224, 3))
                print("WARNING: Frame {} is too small! {}".format(data_list[i].frame_id, max_size))

            data_list[i].faces = [max_face]
            data_list[i].boxes = [max_box]
            data_list[i].ps_roi = ps_roi
            data_list[i].face_ratio = max_size
            # break
            # cv2.imwrite('debug/'+str(i)+'.png', face)
        return data_list

    def process_image(self, file_list, batch_size=16, input_type='image'):
        # def resize_image_with_aspect_ratio(image_path, max_size=640):
        #     # 读取图像
        #     img = cv2.imread(image_path)

        #     # 获取原始图像的高度和宽度
        #     original_height, original_width = img.shape[:2]

        #     # 计算resize的比例，选择较小的比例以确保长宽都不超过最大阈值
        #     ratio = min(max_size / original_height, max_size / original_width)

        #     # 新的高度和宽度
        #     new_height = int(original_height * ratio)
        #     new_width = int(original_width * ratio)

        #     # 使用cv2.resize按比例resize图像
        #     resized_img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)

        #     return resized_img

        img_list = []
        for i in range(min(batch_size, len(file_list))):
            
            # FileStd
            file = file_list[i]
            # match = re.search(r'_(\d+).png$', image_paths[i])

            # if match:
            #     frame_number = int(match.group(1))
            #     # print("帧数:", frame_number)
            # else:
            #     frame_number = 0
            #     # if input_type == 'video':
            #     #     print("WARNING:未找到匹配的数字!", image_paths[i])
            if file.content is None:
                print('Got NoneType content, continue...')
                continue
            img_list.append(ImageStd(file))
            # img_list = resize_image_with_aspect_ratio(content)
            # if input_type != 'video':
            #     img = cv2.resize(img, (640, 640), cv2.INTER_CUBIC)
            # raw_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # frame_num.append(frame_number)
            # frame_map[frame_number] = image_paths[i]

            # img = np.float32(img)
            # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)

            # img -= (104, 117, 123)
            # img = img.transpose(2, 0, 1)
            # img = torch.from_numpy(img)
            # img_list.append(img)
            # scale_list.append(scale)

        file_list = self.face_detection(img_list)
        
        return file_list

    def preprocess(self, data_path, batch_size=64, save_path=None, for_mask=False):
        def read_from_path(image_path):
            raw_img = cv2.imread(image_path)
            img = np.float32(raw_img)
            
            img = cv2.resize(img, (640, 640), cv2.INTER_CUBIC)
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            return img, scale, raw_img
        if not for_mask:
            image_paths = glob.glob(os.path.join(data_path, '*.jpg')) + glob.glob(os.path.join(data_path, '*.png'))
        else:
            image_paths = [data_path]
        batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

        for i, batch in enumerate(batches):
            imgs, scales, raw_imgs = [], [], []
            for j in range(len(batch)):
                try:
                    img, scale, raw_img = read_from_path(batch[j])
                    imgs.append(img)
                    scales.append(scale)
                    raw_imgs.append(raw_img)
                except:
                    continue
            x = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                loc, conf, landms = self.net(x)

            for j in range(len(imgs)):
                frame_id = batch_size*i+j
                faces, bboxes = self.get_face(raw_imgs[j], loc[j], conf[j], landms[j], scales[j])

                # print(bboxes)
                qualified_faces = []
                max_face, max_size, max_box = None, -1, None
                for face, box in zip(faces, bboxes):
                    width, height = box[2] - box[0], box[3] - box[1]
                    # if width * height < self.limit_size:
                    #     continue
                    area = width * height
                    if area > max_size:
                        max_size = area
                        max_face = face
                        max_box = box
                    qualified_faces.append((face, box))
                
            if not for_mask:
                if len(qualified_faces) == 0:
                    # FIXME: bad choice
                    max_face = np.zeros((224, 224, 3))
                    bboxes = [[0,0,0,0]]
                    print("WARNING: Frame {} has no qualified face!".format(frame_id))
                else: cv2.imwrite(save_path+'aa' + str(frame_id)+'.png', max_face)
            else:
                return max_box
                
            
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

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


import glob
if __name__ == '__main__':
    config = {}
    # config['local_rank'] = args.local_rank
    # print(args.local_rank)
    # torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')

    test = Inference(config)
    directory = '/data/huangwen/worklab/AdaptiveCross/datasets/ps'
    image_paths = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.png'))
    print(image_paths)
    # test.process_image(image_paths=image_files)
    # test.infer(image_paths)

