# 图像 / 视频伪造检测（支持华为昇腾推理部署）

**华为昇腾（Huawei Ascend）支持：已实现基于 CANN / ais_bench 的 OM 离线模型推理适配，支持图片、视频及 socket 服务部署。**

| 支持项 | 当前范围 |
| --- | --- |
| 默认适配目标 | Atlas 300I Pro / Ascend310P3，CANN 8.0.0 正式版 |
| 推理模式 | 单进程单 NPU、静态 float32 NCHW OM；设备号可配置 |
| 检测流程 | RetinaFace 人脸检测、RefineNet 换脸检测、PSCC 篡改定位与分类 |
| 验证状态 | 系统此前已完成华为昇腾部署验证（项目方确认）；本次应用层自动测试通过 |
| 训练支持 | 当前未支持昇腾训练 |

本项目包含 RetinaFace 人脸检测、RefineNet 换脸检测、PSCC 篡改定位与分类。
已有昇腾 OM 资产，但原工程不是可完整复现的训练与部署发行包。本次补齐了可配置的
单卡 OM 推理、图片 / 视频入口、socket 服务、编译工具和自动测试。
详细缺项见 [项目审计](docs/PROJECT_AUDIT.md)。

## 获取代码与模型

```bash
git clone https://github.com/HF-biu/shengteng-deepfake-detect.git
cd shengteng-deepfake-detect
```

Git 仓库包含源码、配置、测试及文档。模型权重、ONNX / OM 文件、业务图片与运行结果
作为独立部署资产管理，不随源码上传。请从项目维护方获取模型，并按
[模型资产清单](docs/MODEL_ASSETS.json)中的相对路径放置；清单提供文件大小和 SHA-256 校验值。
默认推理需要根目录的 `FaceDetector.om`、`pscc.om`、`refine_128.om`；重新编译还需
`FaceDetector.onnx`、`pscc_net.onnx`、`cfl_128.onnx`。可选热力图需要 `weights/refine_cfl.pth`。

历史业务脚本 `ais_test.py`、`fetch_and_download.py`、`run.sh` 使用 `MEDIA_API_URL`
环境变量指定数据接口地址，运行前需配置；核心昇腾推理入口不依赖该业务接口。

## 部署基线

工程默认目标为 **Atlas 300I Pro / Ascend310P3 + CANN 8.0.0 正式版**。
这是针对现有模型年代与目标芯片选择的保守基线，并非“市场占有率最高”或“所有环境最稳定”的保证。
目标系统为 Linux；Python 建议 3.10，主机架构可为 x86_64 / aarch64，具体操作系统、
驱动、固件必须使用设备厂商和 CANN 配套表允许的组合。部署前在
[华为资源下载与版本配套查询](https://www.hiascend.com/developer/download)核对。

仓库已有 OM 内嵌 ATC 命令显示 `Ascend310P3`、`8.0.RC1/aarch64-linux`。
新部署建议在选定的 CANN 8.0.0 环境重新编译并验证，不能仅以“存在 OM 文件”判断兼容。
参考 [CANN 8.0.0 ATC 文档](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0039.html)。
使用 310B 或 910B 时必须改用实际 `soc_version` 重新编译，并采用对应型号的软件配套。

## 安装

先在服务器安装匹配的驱动 / 固件、CANN toolkit 和 310P 算子包，确认 `npu-smi info` 正常。
下面命令从项目根目录执行；CANN 安装路径按实际环境调整。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

从 [Ascend msit benchmark 安装说明](https://gitee.com/ascend/msit/blob/master/msit/components/benchmark/README.md)
获取与 CANN、Python 和 CPU 架构配套的 **aclruntime 与 ais_bench 推理包**，将两个 wheel 放在 `wheels/` 后安装：

```bash
python -m pip install wheels/aclruntime-*.whl wheels/ais_bench-*.whl
python -c "from ais_bench.infer.interface import InferSession; print('Ascend runtime import OK')"
```

这里使用的是提供 `ais_bench.infer.interface` 的 OM 推理工具，避免安装同名但用途不同的评测包。
运行时 API 依据 [官方 InferSession 文档](https://gitee.com/ascend/msit/blob/master/msit/docs/benchmark/API_GUIDE.md)。
核心应用不依赖 CUDA、torch_npu 或 PyTorch；所选 ais_bench wheel 自身的依赖以官方包为准。

## 编译和验证模型

```bash
python -m pip install -r requirements-dev.txt
python compile_ascend.py --soc-version Ascend310P3 --dry-run
python compile_ascend.py --soc-version Ascend310P3 --output-dir build/ascend310p3
python ascend_infer.py --config build/ascend310p3/inference.yaml --device-id 0 --smoke-test
```

编译器从 ONNX 读取真实输入名，检查固定维度，调用 ATC，并生成含绝对模型路径的配置和命令清单。
它不覆盖仓库原模型；目标目录已有同名 OM 时拒绝覆盖，重编译请换一个输出目录。
生成配置在服务器上就地使用；迁移目录后需更新配置中的绝对路径。

| 环节 | 编译源 | 仓库现有 OM | 输入 |
| --- | --- | --- | --- |
| 人脸检测 | `FaceDetector.onnx` | `FaceDetector.om` | `[64,3,640,640]` |
| PSCC | `pscc_net.onnx` | `pscc.om` | `[64,3,256,256]` |
| 换脸检测 | **`cfl_128.onnx`** | **`refine_128.om`** | `[128,3,224,224]` |

`refine_128_v1.om` 来自另一个 `refine_128.onnx`，不默认替换当前模型，以免改变预测结果。
如果目标 CANN 编译失败，应保留 ATC 日志并定位算子。

也可先验证仓库原 OM：

```bash
python ascend_infer.py --device-id 0 --smoke-test
```

冒烟测试会实际加载并执行三个模型，检查输出非空且有限，并输出张量形状。
它验证可执行性，**不验证检测精度**。上线前需用有标签样本复核检测率、误报率、延迟及内存占用，
并尽可能与训练框架输出逐样本对比。

## 图片、视频与服务

```bash
python ascend_infer.py --config build/ascend310p3/inference.yaml --input example.jpg --output results/image.json
python ascend_infer.py --config build/ascend310p3/inference.yaml --input example.mp4 --input-type video --output results/video.json
python start_service.py --config build/ascend310p3/inference.yaml --device-id 0 --host 0.0.0.0 --port 54321
```

将示例文件名换成实际文件。JSON 包含 `frame_id`、`state`、`face_detected`、换脸概率、
篡改分类概率、掩码比例和兼容旧逻辑的融合 `score=max(换脸概率,掩码比例)`。
`score` 是规则融合分数，未校准为概率。无合格人脸时延续原模型流程的零分占位；
应结合 `face_detected=false` 判断为未获得有效人脸结果，不能据此证明内容真实。
视频逐批处理所有可解码帧；当前输出逐帧结果，不包含时间段合并。

socket 协议保留原格式：

```text
Content-Length: <文件字节数>\nInput-Type: image\n\n\n\n<文件原始字节>
```

视频使用 `Input-Type: video`。成功返回数字字符串（视频取各帧融合分数的最大值），
失败返回 `{"error":"..."}`，随后关闭连接。当前服务串行处理请求，让 ACL 会话一直在创建它的线程使用；
多卡部署可用不同 `--device-id`、`--port` 启动多个进程。
读取超时为 60 秒，单请求文件上限 256 MiB。

## 配置与范围

`config/inference.yaml` 中 `ascend.device_id` 统一控制三个模型，默认 0。
相对模型路径按项目根目录解析。实际静态批次从 OM 元数据读取；`batch_size` 为应用切分大小，
必须为正整数且不超过 OM 编译批次。末批自动补零，各模型输出按真实数量裁剪。
降低切分大小不会降低 OM 本身的固定批次显存需求；改变编译批次需要兼容的 ONNX 导出。

可选热力图：安装相互兼容的 CPU PyTorch / torchvision、`requirements-cam.txt` 中依赖，
设置 `ascend.enable_cam: true`。权重采用严格加载；如与源码不一致会明确失败。
默认关闭时返回原人脸图作为旧接口兼容占位，不生成解释性热力图。
当前 JSON / socket 入口只返回数值；自定义图片展示可读取底层 `Inference.infer()` 的 `cam_img`。

本轮支持范围是单进程单 NPU 静态 float32 NCHW OM 推理。
**当前不支持昇腾训练**：`train.py` 仍是旧实验脚本，训练数据、部分权重与入口均不完整。
旧 `inference.py`、`run_test.py`、`ais_infer_v1.py`、`ais_test.py` 保留作历史实验参考，
部署请使用本 README 的入口。`run.sh` 是历史业务接口调用，不是启动脚本。

## 测试

```bash
python -m pytest -q
```

测试用模拟 InferSession 验证设备号、静态批次、尾批、空输入、无脸、输出形式、配置阈值、
视频分批、资源释放、socket 断连与请求处理；不需要 NPU。
本次 Windows 主机的旧 Python 3.8 / NumPy DLL 在 pytest 捕获环境初始化失败，使用以下方式完成验证：

```bash
python -c "import numpy,cv2; import pytest; raise SystemExit(pytest.main(['-q','-s','-p','no:faulthandler']))"
```

本机 **31 项测试通过**，Python 源文件语法检查通过。
系统此前的华为昇腾部署验证已由项目方确认。
