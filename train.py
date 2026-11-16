import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import findLastCheckpoint, save_image, adjust_learning_rate
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args
from utils.load_tdata import TrainData
from utils.load_tdata import ValData

# device = torch.device('cuda:0')
# device_ids = [0]

device='cpu'
def train(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    # load train data
    train_data_loader = DataLoader(TrainData(args), batch_size=args['train_bs'], shuffle=True, num_workers=8)
    
    FENet = FENet.to(device)
    SegNet = SegNet.to(device)
    ClsNet = ClsNet.to(device)

    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=float(args['learning_rate']))

    FENet = nn.DataParallel(FENet)
    SegNet = nn.DataParallel(SegNet)
    ClsNet = nn.DataParallel(ClsNet)

    FENet_dir = './ckpt/{}_checkpoint'.format(FENet_name)
    if not os.path.exists(FENet_dir):
        os.mkdir(FENet_dir)

    SegNet_dir = './ckpt/{}_checkpoint'.format(SegNet_name)
    if not os.path.exists(SegNet_dir):
        os.mkdir(SegNet_dir)

    ClsNet_dir = './ckpt/{}_checkpoint'.format(ClsNet_name)
    if not os.path.exists(ClsNet_dir):
        os.mkdir(ClsNet_dir)

    # load FENet weight
    try:
        FENet_weight_path = '{}/{}.pth'.format(FENet_dir, FENet_name)
        FENet_state_dict = torch.load(FENet_weight_path, map_location='cpu')
        FENet.load_state_dict(FENet_state_dict)
        print('{} weight-loading succeeds: {}'.format(FENet_name, FENet_weight_path))
    except Exception as e:
        print('{}. {} weight-loading fails'.format(e, FENet_name))

    # load SegNet weight
    try:
        SegNet_weight_path = '{}/{}.pth'.format(SegNet_dir, SegNet_name)
        SegNet_state_dict = torch.load(SegNet_weight_path, map_location='cpu')
        SegNet.load_state_dict(SegNet_state_dict)
        print('{} weight-loading succeeds: {}'.format(SegNet_name, SegNet_weight_path))
    except:
        print('{} weight-loading fails'.format(SegNet_name))

    # load ClsNet weight
    try:
        ClsNet_weight_path = '{}/{}.pth'.format(ClsNet_dir, ClsNet_name)
        ClsNet_state_dict = torch.load(ClsNet_weight_path, map_location='cpu')
        ClsNet.load_state_dict(ClsNet_state_dict)
        print('{} weight-loading succeeds: {}'.format(ClsNet_name, ClsNet_weight_path))
    except:
        print('{} weight-loading fails'.format(ClsNet_name))

    # validation
    print('length of traindata: {}'.format(len(train_data_loader)))
    previous_score = validation(FENet, SegNet, ClsNet, args)
    print('previous_score {0:.4f}'.format(previous_score))

    # cross entropy loss
    authentic_ratio = args['train_ratio'][0]
    fake_ratio = 1 - authentic_ratio
    print('authentic_ratio: {}'.format(authentic_ratio), 'fake_ratio: {}'.format(fake_ratio))
    weights = [1. / authentic_ratio, 1. / fake_ratio]
    weights = torch.tensor(weights)
    CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    BCE_loss_full = nn.BCELoss(reduction='none').to(device)

    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
    print(initial_epoch)
    if initial_epoch > 0:
        try:
            FENet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, initial_epoch), 'cpu')
            FENet.load_state_dict(FENet_checkpoint['model'])
            print("resuming FENet by loading epoch {}".format(initial_epoch))

            SegNet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, initial_epoch), 'cpu')
            SegNet.load_state_dict(SegNet_checkpoint['model'])
            print("resuming SegNet by loading epoch {}".format(initial_epoch))

            ClsNet_checkpoint = torch.load(
                '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, initial_epoch), 'cpu')
            ClsNet.load_state_dict(ClsNet_checkpoint['model'])
            optimizer.load_state_dict(ClsNet_checkpoint['optimizer'])
            print("resuming ClsNet by loading epoch {}".format(initial_epoch))
        except Exception as e:
            print(e)
            print('cannot load checkpoint on epoch {}'.format(initial_epoch))
            initial_epoch = 0
            print("resuming by loading epoch {}".format(initial_epoch))

    for epoch in range(0, args['num_epochs']):

        adjust_learning_rate(optimizer, epoch, args['lr_strategy'], args['lr_decay_step'])
        seg_total, seg_correct, seg_loss_sum = [0] * 3
        cls_total, cls_correct, cls_loss_sum = [0] * 3

        for batch_id, train_data in enumerate(train_data_loader):

            image, masks, cls = train_data
            cls[cls != 0] = 1

            mask1, mask2, mask3, mask4 = masks

            # median-frequency class weighting
            mask1_balance = torch.ones_like(mask1)
            if (mask1 == 1).sum():
                mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
                mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())
            else:
                print('Mask1 balance is not working!')

            mask2_balance = torch.ones_like(mask2)
            if (mask2 == 1).sum():
                mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
                mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())
            else:
                print('Mask2 balance is not working!')

            mask3_balance = torch.ones_like(mask3)
            if (mask3 == 1).sum():
                mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
                mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())
            else:
                print('Mask3 balance is not working!')

            mask4_balance = torch.ones_like(mask4)
            if (mask4 == 1).sum():
                mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
                mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())
            else:
                print('Mask4 balance is not working!')

            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            mask1_balance, mask2_balance, mask3_balance, mask4_balance, = mask1_balance.to(device), mask2_balance.to(
                device), mask3_balance.to(device), mask4_balance.to(device)

            cls = cls.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # feature extraction network
            FENet.train()
            feat = FENet(image)
            SegNet.train()
            [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = SegNet(feat)

            ClsNet.train()
            pred_logit = ClsNet(feat)

            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_mask1.squeeze(dim=1), pred_mask2.squeeze(
                dim=1), pred_mask3.squeeze(dim=1), pred_mask4.squeeze(dim=1)

            mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
            mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
            mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
            mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
            seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

            cls_loss = CE_loss(pred_logit, cls)

            loss = seg_loss + cls_loss

            loss.backward()
            optimizer.step()

            # localization
            binary_mask1 = torch.zeros_like(pred_mask1)
            binary_mask1[pred_mask1 > 0.5] = 1
            binary_mask1[pred_mask1 <= 0.5] = 0

            seg_correct += (binary_mask1 == mask1).sum().item()
            seg_total += int(torch.ones_like(mask1).sum().item())

            # detection
            _, binary_cls = torch.max(pred_logit, 1)

            cls_correct += (binary_cls == cls).sum().item()
            cls_total += int(torch.ones_like(cls).sum().item())

            # print statistics
            seg_loss_sum += seg_loss.item()
            cls_loss_sum += cls_loss.item()

            if batch_id % 5 == 0:
                print(
                    '[{0}, {1}] batch_loc_acc: [{2}/{3}] {4:.2f}, seg_loss: {5:.4f}; batch_cls_acc: [{6}/{7}] {8:.2f}, '
                    'cls_loss: {9:.4f}'.format(epoch + 1, batch_id + 1, seg_correct, seg_total,
                                               seg_correct / seg_total * 100,
                                               seg_loss_sum / 100, cls_correct, cls_total,
                                               cls_correct / cls_total * 100,
                                               cls_loss_sum / 100))

                seg_total, seg_correct, seg_loss_sum, cls_total, cls_correct, cls_loss_sum = [0] * 6

        if epoch % 20 == 0:
            FENet_checkpoint = {'model': FENet.state_dict(),
                                'optimizer': optimizer.state_dict()}
            torch.save(FENet_checkpoint,
                       '{0}/{1}_{2}_test.pth'.format(FENet_dir, FENet_name, epoch + 1))

            SegNet_checkpoint = {'model': SegNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(SegNet_checkpoint,
                       '{0}/{1}_{2}_test.pth'.format(SegNet_dir, SegNet_name, epoch + 1))

            ClsNet_checkpoint = {'model': ClsNet.state_dict(),
                                 'optimizer': optimizer.state_dict()}
            torch.save(ClsNet_checkpoint,
                       '{0}/{1}_{2}_test.pth'.format(ClsNet_dir, ClsNet_name, epoch + 1))

            current_score = validation(FENet, SegNet, ClsNet, args)
            print('current_score: {0:.4f}'.format(current_score))

            if current_score >= previous_score:
                torch.save(FENet.state_dict(), '{0}/test_{1}.pth'.format(FENet_dir, FENet_name))
                torch.save(SegNet.state_dict(), '{0}/test_{1}.pth'.format(SegNet_dir, SegNet_name))
                torch.save(ClsNet.state_dict(), '{0}/test_{1}.pth'.format(ClsNet_dir, ClsNet_name))
                previous_score = current_score

def validation(FENet, SegNet, ClsNet, args):
    val_data_loader = DataLoader(ValData(args), batch_size=args['val_bs'], shuffle=False, num_workers=8)

    pred_soft_ncls = []
    ncls = []
    seg_correct, seg_total, cls_correct, cls_total = [0] * 4

    for batch_id, val_data in enumerate(val_data_loader):

        image, mask, cls, name = val_data

        image = image.to(device)
        mask = mask.to(device)
        cls[cls != 0] = 1
        cls = cls.to(device)

        with torch.no_grad():

            # feature extraction network
            FENet.eval()
            feat = FENet(image)

            # localization network
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            # classification network
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        if pred_mask.shape != mask.shape:
            pred_mask = F.interpolate(pred_mask, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)

        binary_mask1 = torch.zeros_like(pred_mask)
        binary_mask1[pred_mask > 0.5] = 1
        binary_mask1[pred_mask <= 0.5] = 0

        seg_correct += (binary_mask1 == mask).sum().item()
        seg_total += int(torch.ones_like(mask).sum().item())

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        pred_soft_ncls.extend(pred_logit[:, 1])
        ncls.extend(cls)

        _, binary_cls = torch.max(pred_logit, 1)
        cls_correct += (binary_cls == cls).sum().item()
        cls_total += int(torch.ones_like(cls).sum().item())

        if args['save_tag']:
            save_image(pred_mask, name, 'mask')
    return (seg_correct / seg_total + cls_correct / cls_total) / 2


import cv2
import yaml
import glob
import shutil
import numpy as np
from inference import Processer

def save_image_paths_to_file(target_path, output_file='./pre_data/ps_train/authentic/authentic.txt', pre_path='authentic'):
    
    if not output_file.lower().endswith('.txt'):
        output_file += '.txt'
    
    # 获取当前工作目录
    current_directory = os.getcwd()
    
    # 如果目标路径不存在，则退出
    if not os.path.exists(target_path):
        print(f"目标路径 '{target_path}' 不存在。")
        return
    
    # 遍历目标路径中的所有文件
    image_paths = []
    for root, dirs, files in os.walk(target_path):
        for filename in files:
            # 检查文件扩展名是否为图片格式
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # 构建相对路径
                # relative_path = os.path.relpath(os.path.join(root, filename), current_directory)
                relative_path = os.path.join(pre_path, filename) 
                image_paths.append(relative_path)
    
    # 保存图像路径到文件
    with open(output_file, 'w') as file:
        for path in image_paths:
            file.write(path + '\n')
    
    print(f"图像路径已保存到 '{output_file}'。")

def copy_images(src_folder, dest_folder):
    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 构建完整的文件路径
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)
            # 复制图片
            shutil.copy2(src_file, dest_file)

from tqdm import tqdm
def preprocess(cfg):
    
    # --------------修改这里-------------
    # source data path, 这里放没有检测彩信换脸出来的数据
    real_data_path = cfg['source real data']
    fake_data_path = cfg['source fake data']
    save_path = cfg['save path']
    # ----------------------------------
    if not os.path.exists(os.path.join(save_path, 'authentic/authentic')):
        os.makedirs(os.path.join(save_path, 'authentic/authentic'))
    if not os.path.exists(os.path.join(save_path, 'splice')):
        os.makedirs(os.path.join(save_path, 'splice'))
        os.makedirs(os.path.join(save_path, 'splice/fake'))
        os.makedirs(os.path.join(save_path, 'splice/mask'))
    copy_images(real_data_path, os.path.join(save_path, 'authentic/authentic'))
    print('----------------')
    print('Authentic images processed.')
    print('----------------')
    def get_crop_region(det, origin, type='dlib'):
        origin_h, origin_w, _ = origin
        if type == 'dlib': left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
        else: left, top, right, bottom = det
        center_w, center_h = (left + right) // 2, (top + bottom) // 2
        w_stx, h_stx = max(0, center_w-128), max(0, center_h-128)
        w_end, h_end = min(origin_w, w_stx+256), min(origin_h, h_stx+256)
        return (w_stx, h_stx, w_end, h_end)

    def count_images_in_folder(folder_path):
        image_count = 0
        
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' doesn't exist")
            return 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
                
        return image_count


    
    processer = Processer(cfg, 'cpu')


    # count_idx = count_images_in_folder(save_path)
    count_idx = 0

    img_files = glob.glob(os.path.join(fake_data_path, '*.jpg')) + glob.glob(os.path.join(fake_data_path, '*.png')) + glob.glob(os.path.join(fake_data_path, '*.jpeg')) 

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('./dlib_tools/shape_predictor_81_face_landmarks.dat')


    for img in tqdm(img_files, total=len(img_files)):

        image = cv2.imread(img)
        if image is None:
            print(f"读取图像{img}失败，请检查文件路径是否正确，或者文件是否损坏。")
            continue
        # 根据dlib其示例显示，需要转为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        has_face = True
        # faces = detector(gray)
        # if faces:
        #     face = max(faces, key=lambda rect: rect.width() * rect.height())
        #     shape = predictor(image, face)
        #     landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        #     fill_h = landmarks[8][1] - landmarks[5][1]
        #     for i in range(68, 76):
        #         landmarks[i][1] = max(0, landmarks[i][1]-fill_h)
        #     landmarks[80][1] = max(0, landmarks[80][1]-fill_h)

        #     center = np.mean(landmarks, axis=0)
            
        #     enlarged_points = []
        #     for point in landmarks:
        #         vector = point - center
        #         enlarged_point = center + 1.2 * vector
        #         enlarged_points.append(enlarged_point)
        #     enlarged_points = np.array(enlarged_points, dtype=np.int32)

        #     mask = np.zeros_like(image, dtype=np.uint8)
            
        #     hull = cv2.convexHull(enlarged_points)

        #     cv2.fillConvexPoly(mask, hull, (255, 255, 255))
            
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     crop_region = get_crop_region(face, image.shape, 'dlib')
        # else: 
        box = processer.preprocess(img, batch_size=1, for_mask=True)
        if box is None: has_face = False
        else:
            mask = np.zeros_like(image, dtype=np.uint8)
            w = box[2] - box[0]
            h = box[3] - box[1]
            dw = int(0.1 * w)
            dh = int(0.1 * h)
            x1_new = max(box[0] - dw, 0)
            y1_new = max(box[1] - dh, 0)
            x2_new = min(box[2] + dw, image.shape[1])
            y2_new = min(box[3] + dh, image.shape[0])

            mask[y1_new:y2_new, x1_new:x2_new] = 255
            # mask[box[1]:box[3], box[0]:box[2]] = 255
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # w_stx, h_stx, w_end, h_end
            crop_region = get_crop_region(box, image.shape, 'retina')

        if has_face == True:
            crop_face = image[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
            crop_mask = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
            crop_face = cv2.resize(crop_face, (256, 256))
            crop_mask = cv2.resize(crop_mask, (256, 256))
            save_path_face = os.path.join(save_path, f'splice/fake/{str(count_idx)}.jpg')
            save_path_mask = os.path.join(save_path, f'splice/mask/{str(count_idx)}.jpg')
            crop_face = cv2.cvtColor(crop_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path_face, crop_face)
            cv2.imwrite(save_path_mask, crop_mask)
            count_idx += 1
        else:
            print(f'No face. Path:{img}')

    save_image_paths_to_file(os.path.join(save_path, 'authentic'), output_file=os.path.join(save_path, 'authentic/authentic.txt'), pre_path='authentic')
    save_image_paths_to_file(os.path.join(save_path, 'splice'), output_file=os.path.join(save_path, 'splice/fake.txt'), pre_path='fake')


if __name__ == '__main__':
    args = get_pscc_args()
    with open('./config/inference.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    args['path'] = cfg['save path']
    preprocess()
    train(args)