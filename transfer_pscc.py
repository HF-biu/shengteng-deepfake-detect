import torch
import torch.onnx
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args

args = get_pscc_args()
FENet_cfg = get_hrnet_cfg()
FENet = get_seg_model(FENet_cfg)
SegNet = NLCDetection(args)
ClsNet = DetectionHead(args)

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


# FENet_checkpoint = torch.load("./good_ckpt/20240705-2/test_HRNet.pth")
# FENet.load_state_dict(remove_prefix(FENet_checkpoint, 'module.'))
# SegNet_checkpoint = torch.load("./good_ckpt/20240705-2/test_NLCDetection.pth")
# SegNet.load_state_dict(remove_prefix(SegNet_checkpoint, 'module.'))
# ClsNet_checkpoint = torch.load("./good_ckpt/20240705-2/test_DetectionHead.pth")
# ClsNet.load_state_dict(remove_prefix(ClsNet_checkpoint, 'module.'))


FENet_checkpoint = torch.load("./good_ckpt/20240716/HRNet_checkpoint/HRNet_21_test.pth")
FENet.load_state_dict(remove_prefix(FENet_checkpoint['model'], 'module.'))
# FENet.load_state_dict(FENet_checkpoint)

SegNet_checkpoint = torch.load("./good_ckpt/20240716/NLCDetection_checkpoint/NLCDetection_21_test.pth")
SegNet.load_state_dict(remove_prefix(SegNet_checkpoint['model'], 'module.'))
# SegNet.load_state_dict(SegNet_checkpoint)

ClsNet_checkpoint = torch.load("./good_ckpt/20240716/DetectionHead_checkpoint/DetectionHead_21_test.pth")
ClsNet.load_state_dict(remove_prefix(ClsNet_checkpoint['model'], 'module.'))
# ClsNet.load_state_dict(ClsNet_checkpoint)


# FENet = ...  # 加载 FENet
# SegNet = ...  # 加载 SegNet
# ClsNet = ...  # 加载 ClsNet

image = torch.randn(1, 3, 256, 256) 

FENet.eval()
SegNet.eval()
ClsNet.eval()

class PSCCNet(torch.nn.Module):
    def __init__(self, FENet, SegNet, ClsNet):
        super(PSCCNet, self).__init__()
        self.FENet = FENet
        self.SegNet = SegNet
        self.ClsNet = ClsNet

    def forward(self, image):
        feat = self.FENet(image)
        pred_mask1, pred_mask2, pred_mask3, pred_mask4 = self.SegNet(feat)
        pred_logit = self.ClsNet(feat)
        return pred_mask1, pred_mask2, pred_mask3, pred_mask4, pred_logit


pscc_net = PSCCNet(FENet, SegNet, ClsNet)

# 转换为 ONNX
torch.onnx.export(
    pscc_net,               # 模型
    image,                  # 示例输入
    "pscc_net.onnx",        # 输出文件名
    export_params=True,     # 导出模型参数
    opset_version=11,       # ONNX opset 版本
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=['input'],  # 输入节点名称
    output_names=['pred_mask1', 'pred_mask2', 'pred_mask3', 'pred_mask4', 'pred_logit'],  # 输出节点名称
    dynamic_axes={
        'input': {0: 'batch_size', 2:"Height", 3:"Width"},  # 动态维度
        'pred_mask1': {0: 'batch_size', 2:"Height", 3:"Width"},
        'pred_mask2': {0: 'batch_size', 2:"Height", 3:"Width"},
        'pred_mask3': {0: 'batch_size', 2:"Height", 3:"Width"},
        'pred_mask4': {0: 'batch_size', 2:"Height", 3:"Width"},
        'pred_logit': {0: 'batch_size'}
    }
)

print("Model has been converted to ONNX format")
