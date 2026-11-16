# 项目完整性与昇腾适配审计

审计日期：2026-11-16。对象：当前目录中的源码、配置、模型文件和实验脚本。

## 判断

项目具备图像 / 视频伪造检测的主要网络与推理模型，已有真实的昇腾离线部署工作，
但不具备完整、可重复的训练到生产交付闭环。项目方确认系统此前已完成华为昇腾部署验证；
本次修复应用层推理部署链路并补充自动回归测试。

## 已有资产证据

- `models/retinaface.py`、`models/seg_hrnet.py`、`models/NLCDetection.py`、
  `models/detection_head.py` 和 `model/` 包含网络实现。
- `weights/` 中有 `refine_cfl.pth` 与 `ckpt_best.pth`；`good_ckpt/20240716/` 有三个 PSCC 子模型的检查点。
- 根目录有四个 ONNX 和四个 OM 文件。三个默认 OM 的内嵌 ATC 命令明确包含
  `--soc_version=Ascend310P3`，工具路径为 `ascend-toolkit/8.0.RC1/aarch64-linux/bin/atc.bin`。
- 内嵌编译映射：`FaceDetector.onnx → FaceDetector.om`，`pscc_net.onnx → pscc.om`，
  **`cfl_128.onnx → refine_128.om`**；另一个 `refine_128.onnx → refine_128_v1.om`。
- 原 `ais_infer.py` 确实调用 `ais_bench`，因此不能将该项目描述为“完全没有昇腾适配”。

## 原推理链路问题及本次处理

| 问题 | 处理 |
| --- | --- |
| RetinaFace / PSCC 固定 device 0，换脸 MultiDeviceSession 用 `t+1`，单卡设备 0 无法独立跑全链 | 三模型共用 `ascend.device_id`，默认 0 |
| 配置批次与 OM 批次未校验，RetinaFace 硬编码 64 | 从 OM 元数据读取真实批次，检查 NCHW / float32，末批补零并裁剪所有输出 |
| 空列表访问最后一个 batch；换脸输出按向量假设，标量输出可能失败 | 空输入返回空结果，支持 `[N]`、`[N,1]`、`[N,2]` |
| 强制导入 torch / timm / grad-cam，OM 业务也被可视化依赖阻断 | CPU 热力图延迟加载，默认关闭 |
| 热力图权重 strict=False 静默忽略不匹配 | 启用后严格加载权重 |
| 阈值从工作目录全局配置读取，传入配置不生效 | 实例从传入配置获取阈值；新入口相对路径稳定 |
| 服务导入旧 CUDA 处理器，导入即加载模型，多线程共享推理状态 | 服务切换 Ascend 入口，启动时加载，串行处理，退出释放会话 |
| socket 在客户端提前断连时可能持续等待 | 检测断连、长度、请求类型和读取超时 |
| 资源无显式释放 | 会话 close / 上下文管理，构造失败释放已加载模型 |
| `sorted()` 结果未赋值；掩码累加使用 uint8 可能溢出 | 原地排序；浮点累加并处理零分母 |
| 默认向工作目录写入调试图 | 删除推理链路的无条件调试输出 |
| 无依赖文件、部署说明、可控 CLI、自动测试 | 新增 README、依赖清单、编译 / 推理 CLI、模拟会话回归测试 |

## 尚不完整的部分

1. **训练无法原样复现**：`train.py` 主入口调用 `preprocess()`，函数却要求 `cfg`；
   读取 `cfg['save path']`，当前配置不存在该字段；数据预处理还要求 `source real data`、
   `source fake data`。配置指定的 `ps_train/` 与训练样本清单未提供。
2. **旧 CUDA 路径损坏或依赖缺失**：`inference.py` 导入 `decode/decode_landm`，
   当前 `utils/box_utils.py` 只有 NumPy 版本；PriorBox 返回 NumPy 而旧路径按 Tensor 使用。
   该路径引用的 `weights/Resnet50_Final.pth`、`weights/resnet_test_model.pth` 不在目录中。
   因此新入口直接使用完整 OM，不依赖该路径。
3. **没有训练 NPU 适配**：旧训练脚本写死 CPU，并使用 DataParallel；部分模型函数仍写死 `.cuda()`。
   本轮未伪装成“改一个 device 字符串就支持训练”。如后续训练，需补齐数据和原始权重，
   在支持训练的硬件上实现 torch_npu、checkpoint 迁移并验证前向 / 反向和收敛。
4. **模型来源与版本清单不足**：原导出过程不完整，缺训练数据版本、指标基准、精度对齐样本和发布记录。
   `ge_check_op.json` 留有历史 `infer_shape_error`，仅能视为历史失败记录，不能证明现有模型必然失效。
5. **旧脚本仍有业务耦合**：`ais_test.py`、`run_test.py` 等有硬编码路径、目录假设或私网业务调用；
   `test_demo.py` 是会发送网络请求的手工脚本，不是单元测试。新增 pytest 配置仅收集 `tests/`。
6. **检测语义限制**：现流程集中在人脸及其附近 ROI。无合格人脸延续旧逻辑的占位输出；
   新 JSON 用 `face_detected` 明确标识。融合分数未校准，不能视为通用图像鉴伪准确率。

上述实验脚本保留，未编造缺失数据或权重。默认支持范围以 README 为准。

## 验证记录

- 系统此前已完成华为昇腾部署验证，由项目方确认。
- 本次自动回归测试环境：Windows，Python 3.8.8，使用模拟 InferSession。
- 31 项测试通过，涵盖模拟 OM 的图片与视频全流程、所有三个模型的一致设备选择、
  batch 0/1/2/3/5、dtype enum、初始化失败清理、scalar / column / logits、socket 接口等。
- 新 CLI 的 `--help` 和全项目 Python AST 语法检查通过。
- 常规 pytest 在本机旧 NumPy DLL 捕获环境中初始化失败，预加载 NumPy / OpenCV 后运行通过，命令见 README。
