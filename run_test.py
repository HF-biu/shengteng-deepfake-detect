import cv2
import numpy as np
import os
import requests
import yaml
import shutil
import time

from urllib.parse import urlparse
from torchvision import transforms as T
from io import BytesIO
import tempfile

from inference import Inference
import socket

ALLOWED_IMG_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'mpeg', 'wmv', 'webm', 'ogg'}


class FileStd():
    def __init__(self, file_type, id, content, frame_id, original_name) -> None:
        self.type = file_type
        self.id = id
        self.content = content
        self.frame_id = frame_id
        self.original_name = original_name
        

class Processer():
    def __init__(self) -> None:
        with open('./config/inference.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        self.infer = Inference(cfg)
        self.input_type = None
        self.fps = None
        self.tic = 0

        self.total = 0
        self.correct = 0
        self.binary_correct = 0

    @staticmethod
    def detect_segments(outputs, threshold=0.1, consecutive_frames=8):
        state_list = outputs.state_list
        

        segments = []
        current_segment = {'start': 0, 'end': 0, 'is_fake': state_list[0][0]}
        consecutive_count = 1
        is_fake = state_list[0][0]
        for i in range(1, len(state_list)):
            i_state = state_list[i][0]
            if i_state == current_segment['is_fake']:
                # 当前帧与前一帧的真假标记相同
                current_segment['end'] = i
                consecutive_count = 1
            else:
                # 当前帧与前一帧的真假标记不同
                consecutive_count += 1
                if consecutive_count >= consecutive_frames:
                    # 连续帧数达到阈值，切换片段
                    segments.append(current_segment)
                    current_segment = {'start': i - consecutive_frames + 1, 'end': i, 'is_fake': i_state}
                    consecutive_count = 1

                    if i_state == 'Fake':
                        is_fake = 'Fake'

        # 添加最后一个段
        segments.append(current_segment)

        return segments, is_fake
    
    def clear(self):
        self.fps = None

    @staticmethod
    def frames_to_time(frames, fps):
        start_seconds = frames['start'] / fps
        end_seconds = frames['end'] / fps

        start_time = '{:02}:{:02}:{:02}'.format(int(start_seconds // 3600), int((start_seconds % 3600) // 60), int(start_seconds % 60))
        end_time = '{:02}:{:02}:{:02}'.format(int(end_seconds // 3600), int((end_seconds % 3600) // 60), int(end_seconds % 60))

        return {'start': start_time, 'end': end_time, 'is_fake': frames['is_fake']}
    
    @staticmethod
    def convert_segments_to_time(segments, fps):
        time_segments = [Processer.frames_to_time(seg, fps) for seg in segments]
        return time_segments

    @staticmethod
    def is_path(var):
        try:
            return os.path.isfile(var) or os.path.isdir(var)
        except:
            return False

    @staticmethod
    def read_img_from_path(image_path):
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    @staticmethod
    def save_image(img, save_dir='./mask/', img_name=None):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # else:
        #     shutil.rmtree(save_dir)
        #     os.makedirs(save_dir)
        existing_files = os.listdir(save_dir)
        img_number = len(existing_files) + 1

        save_path = os.path.join(save_dir, img_name)
        
        img.save(save_path)
        # print(f'Saved {save_path}')

    @staticmethod
    def save_result(save_dir, original_names, content):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        existing_files = os.listdir(save_dir)
        img_name = original_names
        save_path = os.path.join(save_dir, img_name)

        # fmt = '.' + (img_name.split('.'))[-1]
        _, img_encoded = cv2.imencode('.jpg', content)
        img_bytes = img_encoded.tobytes()

        with open(save_path, 'wb') as f:
            f.write(img_bytes)
    
    def process(self, input_path, input_type='image', output_directory_base="", original_names=None):
        self.clear()

        if input_type == 'video':
            if isinstance(input_path, list): input_path = input_path[0]
        # path or bytesIO
        # is_path = self.is_path(input_path[0]) if len(input_path)
        is_path = False
        # is_url = self.is_url(input_path)

        # if is_url:
        #     # Process the image or video from URL
        #     output_folder = self.process_url(input_path)
        # else:
            # Process the image or video from local file path
        self.tic = time.time()
        # file_extension = self.get_file_extension(input_path)

        files_list = []

        self.input_type = input_type
        
        if input_type == 'image':
        # if file_extension in ALLOWED_IMG_EXTENSIONS:
            # output_folder = self.process_image(input_path)
            # use id == -1 to represent images
            if is_path:
                input_imgs = [self.read_img_from_path(path) for path in input_path]
                for i, content in enumerate(input_imgs):
                    files_list.append(FileStd('image', i, content, frame_id=0, original_name=original_names[i]))
            else:   # for BytesIO
                for i, content in enumerate(input_path):
                    image_data = np.frombuffer(content.getvalue(), dtype=np.uint8)
                    # 解码图像
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    if image is None:
                        print('Decoded None...')
                        continue
                    files_list.append(FileStd('image', i, image, frame_id=0, original_name=original_names[i]))
        elif input_type == 'video':
        # elif file_extension in ALLOWED_VIDEO_EXTENSIONS:
        
            if is_path:
                readin_image = self.process_video(input_path)
                for i, content in enumerate(readin_image):
                    files_list.append(FileStd('video', 0, content, frame_id=i, original_name=original_names[0]))
                # files_list = [os.path.join(output_folder, file) for file in os.listdir(output_folder)]
            else:  # for BytesIO
                with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmpfile:
                    tmpfile.write(input_path.getbuffer())
                    readin_image = self.process_video(tmpfile.name)
                for i, content in enumerate(readin_image):
                    cv2.imwrite('./temp/'+str(i)+'.jpg', content)
                    files_list.append(FileStd('video', 0, content, frame_id=i, original_name=original_names[0]))

        else:
            print("File format not allowed.")
        print("Input is", self.input_type, len(files_list))
        self.total += len(files_list)
        # max_len = 1024
        # image_files = image_files[:max_len]


        res_fake, res_seg = self.infer.infer(files_list, input_type=self.input_type)
        if self.input_type == 'video':
            for i, key in enumerate(res_fake):
                seg, res = Processer.detect_segments(res_fake[key])
                seg_to_time = Processer.convert_segments_to_time(seg, self.fps)
                print(seg_to_time)
        else:
            print("Deepfake检测结果：", end=' ')
            for idx, key in enumerate(res_fake):
                print(idx, ":", res_fake[key].state_list[0][0], "{:.2f}".format(res_fake[key].state_list[0][1].item()))
                # if res_fake[key].state_list[0][1].item() > 0.8:
                if res_fake[key].state_list[0][1].item() > 0.8:
                    score = res_fake[key].state_list[0][1].item()
                    save_dir = f'{output_directory_base}/deepfake_fake'
                    existing_files = os.listdir(save_dir)
                    score = res_fake[key].state_list[0][1].item()
                    # img_name = files_list[idx].original_name
                    img_name = f'{score}_' + files_list[idx].original_name
                    self.save_result(save_dir, img_name, files_list[idx].content)
                else:
                    save_dir = f'{output_directory_base}/deepfake_real'
                    existing_files = os.listdir(save_dir)
                    score = res_fake[key].state_list[0][1].item()
                    # img_name = files_list[idx].original_name
                    img_name = f'{score}_' + files_list[idx].original_name
                    self.save_result(save_dir, img_name, files_list[idx].content)

            print("检测结果：", end=' ')
            for i in range(len(res_seg)):
                print('PS检测',i,'：', "{:.2f}".format(res_seg[i].cls), "{:.2f}".format(res_seg[i].binary))
            print()
        if input_type == 'image':
            idx = 0
            for seg in res_seg:
                # for debug only...
                # self.save_image(seg.combine_mask, mask_path, original_names[idx])
                # 此处为二分类模型的准确率计算
                # if seg.binary > 0.5:
                #     self.binary_correct += 1
                #     save_dir = f'{output_directory_base}/binary_fake'
                #     existing_files = os.listdir(save_dir)
                #     img_number = len(existing_files) + 1
                #     img_name = f'{original_names[idx]}'
                #     save_path = os.path.join(save_dir, img_name)
                #     cv2.imwrite(save_path, files_list[idx].content)
                # else:
                #     save_dir = f'{output_directory_base}/binary_real'
                #     existing_files = os.listdir(save_dir)
                #     img_number = len(existing_files) + 1
                #     img_name = f'{original_names[idx]}'
                #     save_path = os.path.join(save_dir, img_name)
                #     cv2.imwrite(save_path, files_list[idx].content)
                    
                if seg.cls > 0.5:
                    self.correct += 1
                    save_dir = f'{output_directory_base}/fake'
                    existing_files = os.listdir(save_dir)
                    img_number = len(existing_files) + 1
                    img_name = files_list[idx].original_name
                    self.save_result(save_dir, img_name, files_list[idx].content)

                    mask_path = f'{save_dir}/mask'
                    self.save_image(seg.combine_mask, mask_path, f'{img_number}.jpg')
                else:
                    save_dir = f'{output_directory_base}/real'
                    existing_files = os.listdir(save_dir)
                    img_number = len(existing_files) + 1
                    img_name = files_list[idx].original_name
                    self.save_result(save_dir, img_name, files_list[idx].content)

                    mask_path = f'{save_dir}/mask'
                    self.save_image(seg.combine_mask, mask_path, f'{img_number}.jpg')

                # if (seg.binary + seg.cls) / 2. > 0.5:
                #     save_dir = f'{output_directory_base}/avg_fake'
                #     existing_files = os.listdir(save_dir)
                #     img_number = len(existing_files) + 1
                #     img_name = f'{original_names[idx]}'
                #     save_path = os.path.join(save_dir, img_name)
                #     cv2.imwrite(save_path, files_list[idx].content)
                # else:
                #     save_dir = f'{output_directory_base}/avg_real'
                #     existing_files = os.listdir(save_dir)
                #     img_number = len(existing_files) + 1
                #     img_name = f'{original_names[idx]}'
                #     save_path = os.path.join(save_dir, img_name)
                #     cv2.imwrite(save_path, files_list[idx].content)
                idx += 1
            #     # 检测置信度：seg.cls.item()
            #     # 预测的ps痕迹掩码（Image格式的图像）: seg.combine_mask
            print('Total forward time: {:.4f} avg {:.4f}'.format((time.time() - self.tic), (time.time() - self.tic) / len(files_list)))
            # print(f'correct: {self.correct}, total: {self.total}')
            # # 根据mask计算的准确率
            # print(f'Seg. acc: {self.correct / self.total}')
            # # 根据二分类结果计算的准确率
            # print(f'Binary acc: {self.binary_correct / self.total}')
            # # 平均二者计算的准确率
            # print(f'Avg. acc: {(self.correct + self.binary_correct) / (self.total * 2)}')
            
        # outputs = self.infer.infer(files_list, input_type=self.input_type)
        # if self.input_type == 'video':
        #     for i, key in enumerate(outputs):
                
        #         seg, res = Processer.detect_segments(outputs[key])
        #         seg_to_time = Processer.convert_segments_to_time(seg, self.fps)
        #         print(seg_to_time)

        
        # # deepfake_res, ps_res = process(...)
        # # for i, key in enumerate(deepfake_res):
        # #     face = deepfake_res[key].state_list[0][3]
        # #     cv2.imwrite('/data/huangwen/worklab/AgentViT/temp/'+str(i)+'.jpg', face)
        # # for i, key in enumerate(outputs[0]):
        # #     if outputs[0][key].state_list[0][1].item() > 0.5:
        # #         self.correct += 1
        # idx = 0
        # for seg in outputs[1]:
        #     # 此处为二分类模型的准确率计算
        #     if seg.binary > 0.5:
        #         self.binary_correct += 1
        #         save_dir = f'{output_directory_base}/binary_fake'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)
        #     else:
        #         save_dir = f'{output_directory_base}/binary_real'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)
                
        #     if seg.cls > 0.5:
        #         self.correct += 1
        #         save_dir = f'{output_directory_base}/fake'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)
        #     else:
        #         save_dir = f'{output_directory_base}/real'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)

        #     if (seg.binary + seg.cls) / 2. > 0.5:
        #         save_dir = f'{output_directory_base}/avg_fake'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)
        #     else:
        #         save_dir = f'{output_directory_base}/avg_real'
        #         existing_files = os.listdir(save_dir)
        #         img_number = len(existing_files) + 1
        #         img_name = files_list[idx].original_name
        #         save_path = os.path.join(save_dir, img_name)
        #         cv2.imwrite(save_path, files_list[idx].content)

        #         mask_path = f'{save_dir}/mask'
        #         self.save_image(seg.combine_mask, mask_path, img_name)
        #     idx += 1
        # #     # 检测置信度：seg.cls.item()
        # #     # 预测的ps痕迹掩码（Image格式的图像）: seg.combine_mask
        # print('Total forward time: {:.4f} avg {:.4f}'.format((time.time() - self.tic), (time.time() - self.tic) / len(files_list)))
        # print(f'correct: {self.correct}, total: {self.total}')
        # # 根据mask计算的准确率
        # print(f'Seg. acc: {self.correct / self.total}')
        # # 根据二分类结果计算的准确率
        # print(f'Binary acc: {self.binary_correct / self.total}')
        # # 平均二者计算的准确率
        # print(f'Avg. acc: {(self.correct + self.binary_correct) / (self.total * 2)}')
        # # for i, key in enumerate(outputs[0]):
        # #     face = outputs[0][key].state_list[0][3]
        # #     cv2.imwrite('./images/processed/'+str(i)+'.jpg', face)
        res_list = []   # 伪造图像置信度
        vis_list = []   # 可视化结果，一个tuple (Deepfake可视化，PS检测可视化)
        for key, seg in zip(res_fake, res_seg):
            res_list.append(max(res_fake[key].state_list[0][1].item(), seg.cls))
            vis_list.append((res_fake[key].state_list[0][-1], seg.combine_mask))
        return res_list

        return outputs

    @staticmethod
    def is_url(path):
        # Check if the path is a URL
        return urlparse(path).scheme in ('http', 'https')
    
    @staticmethod
    def get_file_extension(file_path):
        # Get the file extension from the file path
        if isinstance(file_path, list):
            file_path = file_path[0]
        return os.path.splitext(file_path)[1][1:].lower()

    @staticmethod
    def normalize(img):
        """
        Normalize an image.
        """
        img = T.ToTensor()(img)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)


    def process_url(self, url):
        # Fetch content from URL
        response = requests.get(url)

        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()

            if 'image' in content_type:
                # Process image
                self.process_image_from_bytes(url)
            elif 'video' in content_type:
                # Process video
                self.process_video_from_bytes(url)
                self.input_type = 'video'
            else:
                print("Unsupported content type.")
        else:
            print(f"Failed to fetch content from URL. Status code: {response.status_code}")

    def process_image_from_bytes(self, urls):
        def read_from_url(url):
            try:
                # 发送HTTP请求获取图像
                response = requests.get(url)
                
                # 检查请求是否成功
                if response.status_code == 200:
                    # 使用BytesIO将请求的内容转换为字节流
                    image_stream = BytesIO(response.content)
                    
                    # 从字节流中读取图像
                    image = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    print("Failed to retrieve image. Status code:", response.status_code)
                return image
            except Exception as e:
                print("Error:", e)
                return np.zeros((224, 224, 3))
            
        output_folder = 'test_images'
        # 检查文件夹是否存在
        if os.path.exists(output_folder):
            try:
                # 删除文件夹及其内容
                shutil.rmtree(output_folder)
                print(f"The folder {output_folder} has been emptied.")
            except Exception as e:
                print(f"Error emptying the folder: {e}")
        else:
            print(f"The folder {output_folder} does not exist.")
        # 重新创建空文件夹
        os.makedirs(output_folder)

        for i, url in enumerate(urls):
            img = read_from_url(url)
            cv2.imwrite(output_folder+'/'+str(i)+'.png', img)

        return output_folder
        
    def process_video_from_bytes(self, url):
        # Convert video bytes to NumPy array
        # nparr = np.frombuffer(video_bytes, np.uint8)
        video_name = 'url_video'
        output_folder = f"{video_name}_frames"
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        # Decode the video
        cap = cv2.VideoCapture(url)
        # cap.open('appsrc ! videoconvert ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        # cap.open(cv2.CAP_FFMPEG, 0, nparr)

        if not cap.isOpened():
            print("Error opening video stream")
            return
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {self.fps}")

        # 读取并保存每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保存帧到文件夹
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # 释放视频捕获对象
        cap.release()

        print(f"Processed {frame_count} frames. Frames saved to: {output_folder}")
        return output_folder

    def process_image(self, image_paths):
        # Add your logic to process image
        print(f"Processing image: {image_paths}")
        output_folder = 'test_images'
        # 检查文件夹是否存在
        if os.path.exists(output_folder):
            try:
                # 删除文件夹及其内容
                shutil.rmtree(output_folder)
                print(f"The folder {output_folder} has been emptied.")
            except Exception as e:
                print(f"Error emptying the folder: {e}")
        else:
            print(f"The folder {output_folder} does not exist.")
        # 重新创建空文件夹
        os.makedirs(output_folder)

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            cv2.imwrite(output_folder+'/'+str(i)+'.png', img)

        return output_folder
        
    def process_video(self, video_path):
        # Add your logic to process video
        print(f"Processing video: {video_path}")

        # video_name = os.path.splitext(os.path.basename(video_path))[0]
        # output_folder = f"{video_name}_frames"
        # if os.path.exists(output_folder):
        #     shutil.rmtree(output_folder)
        # os.makedirs(output_folder)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 获取视频帧率
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {self.fps}")

        frame_list = []
        # 读取并保存每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保存帧到文件夹
            # frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.png")
            # cv2.imwrite(frame_filename, frame)
            frame_list.append(frame)
            frame_count += 1

            if frame_count >= 4096: break

        # 释放视频捕获对象
        cap.release()

        print(f"Processed {frame_count} frames.")
        return frame_list
    
import glob
import datetime

if __name__ == '__main__':
    # video_path = 'https://v6.huanqiucdn.cn/4394989evodtranscq1500012236/60f4317a3270835015119079404/v.f100830.mp4'
    # directory = './pulled'
    # directory = './2'
    directory = '/data/huangwen/worklab/AgentViT/pre_data/guoman'
    output_directory_base = f'./test-results/test-{datetime.datetime.now()}'
    os.makedirs(f'{output_directory_base}/fake')
    os.makedirs(f'{output_directory_base}/real')
    os.makedirs(f'{output_directory_base}/binary_fake')
    os.makedirs(f'{output_directory_base}/binary_real')
    os.makedirs(f'{output_directory_base}/avg_fake')
    os.makedirs(f'{output_directory_base}/avg_real')

    os.makedirs(f'{output_directory_base}/fake/mask')
    os.makedirs(f'{output_directory_base}/real/mask')
    os.makedirs(f'{output_directory_base}/binary_fake/mask')
    os.makedirs(f'{output_directory_base}/binary_real/mask')
    os.makedirs(f'{output_directory_base}/avg_fake/mask')
    os.makedirs(f'{output_directory_base}/avg_real/mask')

    os.makedirs(f'{output_directory_base}/deepfake_fake')
    os.makedirs(f'{output_directory_base}/deepfake_real')

    image_paths = glob.glob(os.path.join(directory, '*.jpg')) + glob.glob(os.path.join(directory, '*.png')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.PNG')) + glob.glob(os.path.join(directory, '*.JPEG'))
    # for i, path in enumerate(image_paths):
    #     print(i, path)
    # # image_paths = ['/data/huangwen/worklab/AdaptiveCross/datasets/FaceForensics++/original_sequences/youtube/c23/back_groud_frame/002/000.png']
    # image_paths = image_paths[:100]
    # print(image_paths)
    # image_paths = './images/32bf24fc67ebb5983ea3dd5e8598334b.mp4'
    # 视频文件路径
    # video_file ='/data/huangwen/worklab/AgentViT/1.mp4'
    # original_names = [(video_file.split('/'))[-1]]
    # # 打开视频文件
    # cap = cv2.VideoCapture(video_file)

    # # 创建一个空的 BytesIO 对象
    # video_bytesio = BytesIO()

    # # 按帧读取视频并将帧写入 BytesIO 对象
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret:
    #         # 将帧数据写入 BytesIO 对象
    #         video_bytesio.write(cv2.imencode('.jpg', frame)[1].tobytes())
    #     else:
    #         break

    # # 释放视频对象
    # cap.release()

    # # 将 BytesIO 对象指针移到起始位置
    # video_bytesio.seek(0)
    image_bytesios = []

    original_names_pre = [(image_path.split('/'))[-1] for image_path in image_paths]
    original_names = []

    # 循环遍历图像文件列表
    for i, image_file in enumerate(image_paths):

        # 打开图像文件
        with open(image_file, "rb") as f:
            # 读取图像数据
            image_data = f.read()
        
        # 创建 BytesIO 对象并写入图像数据
        image_bytesio = BytesIO(image_data)
        try:
            image = cv2.imdecode(np.asarray(bytearray(image_bytesio.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(e)
            continue
        if image_bytesio is None:
            print(f'Got NoneType input. Continue...')
            continue
        # 将 BytesIO 对象添加到列表中
        image_bytesios.append(image_bytesio)
        original_names.append(original_names_pre[i])



    processer = Processer()
    # results = processer.process(video_bytesio, input_type='video', output_directory_base=output_directory_base, original_names=original_names)


    for i in range(0, len(image_bytesios), 64):
        print(i, i+64, len(image_bytesios))
        results = processer.process(image_bytesios[i:i+64], input_type='image', output_directory_base=output_directory_base, original_names=original_names[i:i+64])
