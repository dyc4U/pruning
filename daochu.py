import torch
import torch.onnx
import numpy as np
model = torch.load('E:/3_work/0_dyc/project/yolov5-prune-main/weight/yolov5n_Lamp_CWD.pt')
model = model['model']
model.float()
model.eval()
# 创建一个与模型输入相同尺寸的随机张量
x = torch.randn(1, 3, 640, 640, requires_grad=True)
# 导出模型
# 假设model是已经加载了权重的YOLOv5模型
# x是一个与模型输入相同尺寸的随机张量，例如：torch.randn(1, 3, 640, 640)

# 导出模型到ONNX
torch.onnx.export(model,               # 运行的模型
                  x,                   # 模型输入（张量）
                  "yolov5s_coco.onnx", # 输出ONNX文件的名称
                  export_params=True,  # 是否导出模型参数
                  opset_version=11,    # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['images'],   # 输入名，通常使用'images'
                  output_names=['boxes', 'confs', 'classes'],  # 输出名，分别对应边界框、置信度和类别
                  dynamic_axes={'images': {0: 'batch_size'},  # 输入的动态批次大小
                                'boxes': {0: 'batch_size'},    # 输出的动态批次大小
                                'confs': {0: 'batch_size'},    # 输出的动态批次大小
                                'classes': {0: 'batch_size'}}) # 输出的动态批次大小


