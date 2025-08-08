```
项目根目录/
├── datasets/
│ └── ball/
│ ├── train/ # 训练集图片和标注
│ ├── val/ # 验证集图片和标注
│ ├── test/ # 测试集图片和标注
│ └── ball.yaml # 数据集配置文件（路径、类别等）
├── docker/ # Docker 配置文件
├── docs/
│ └── mkdocs.yml # 文档构建配置
├── examples/ # 示例脚本
├── results/ # 训练/推理结果（指标图、可视化等）
├── runs/
│ └── detect/
│ └── train/
│ └── weights/
│ ├── best.pt # 最佳训练权重
│ └── best.onnx # 导出的 ONNX 模型
├── tests/ # 测试脚本
├── ultralytics/ # YOLO 核心库代码
├── 01-模型训练.py # 模型训练脚本
├── 02-模型推理.py # 图片推理脚本
├── 03-模型验证.py # 模型精度验证脚本
├── 04-检验播报.py # 推理结果后处理/通知脚本
├── 05-视频检测.py # 视频推理脚本
├── 06-推理加速.py # 模型加速（含 ONNX/TensorRT 处理）
├── yolo11s.pt # 基础模型权重
├── yolo11s-cls.pt # 分类专用模型权重
├── input_video.mp4 # 视频检测输入文件
├── output_video.mp4 # 视频检测输出文件
└── README.md # 项目说明文档

```

