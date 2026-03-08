# FantasyInfer

轻量级 C++ 推理框架，基于 PNNX 计算图，支持常见 CNN 算子与 YOLO、ResNet 等模型推理。

## 特性

- **PNNX 模型格式**：直接加载 `.pnnx.param` + `.pnnx.bin`，无需额外转换
- **算子与层**：ReLU、Sigmoid、Linear、Convolution、MaxPooling、Flatten、Softmax、AdaptiveAvgPooling、Expression 等算子
- **示例模型**：YOLOv5 目标检测、ResNet 分类、简单线性/卷积图
- **拓扑执行**：自动构建计算图拓扑序并顺序执行

## 依赖

- CMake ≥ 3.16
- C++17
- OpenMP、Armadillo、glog、GTest、OpenCV、BLAS/LAPACK

## 构建

```bash
mkdir build && cd build
cmake ..
make -j
```

## 运行测试

```bash
./FantasyInfer
```

按用例过滤运行，例如：

```bash
# 仅运行 YOLO 测试
./FantasyInfer --gtest_filter="test_network.yolov5"

# 仅运行 ResNet 测试
./FantasyInfer --gtest_filter="test_network.resnet"
```

建议在 `build/` 目录下执行 `./FantasyInfer`，或在运行前于当前目录创建 `log` 目录，否则 glog 会报 “Could not create logging file”（日志仍会打印到 stderr）。

YOLO 测试会读取 `model_file/` 下配置的图片与模型，并将检测结果保存为 `output0.jpg` 等。

## 模型文件

推理使用 PNNX 导出的模型：

- **结构**：`xxx.pnnx.param`（文本）
- **权重**：`xxx.pnnx.bin`（二进制）

从 PyTorch 导出示例：

```python
import torch
import pnnx

model = YourModel()
model.eval()
x = torch.randn(1, 3, 224, 224)  # 示例输入
pnnx.export(model, "model_name", x)
# 生成 model_name.pnnx.param 与 model_name.pnnx.bin
```

将生成的两个文件放入 `model_file/` 或修改测试中的路径即可。

## 目录结构

```
FantasyInfer/
├── CMakeLists.txt
├── main.cpp                    # 入口，初始化 glog + 运行 GTest
├── README.md
├── LICENSE
│
├── include/                    # 头文件
│   ├── data/
│   │   ├── tensor.hpp          # 张量
│   │   ├── tensor_utils.hpp
│   │   └── load_data.hpp
│   ├── runtime/
│   │   ├── runtime_ir.hpp      # 计算图构建与 Forward
│   │   ├── runtime_op.hpp      # RuntimeOperator / RuntimeOperand
│   │   ├── runtime_operand.hpp
│   │   ├── runtime_parameter.hpp
│   │   ├── runtime_attr.hpp
│   │   ├── runtime_datatype.hpp
│   │   ├── ir.h                # pnnx Graph/Operator/Operand/Attribute
│   │   └── store_zip.hpp       # bin 权重读取
│   ├── layer/
│   │   └── abstract/
│   │       ├── layer.hpp
│   │       ├── layer_factory.hpp
│   │       ├── param_layer.hpp
│   │       └── non_param_layer.hpp
│   ├── parser/
│   │   └── parser_expression.hpp
│   ├── utils/
│   │   ├── time/
│   │   └── math/
│   └── status_code.hpp
│
├── source/                     # 实现
│   ├── runtime_ir.cpp          # 图加载、Build、Forward、拓扑排序
│   ├── runtime_op.cpp
│   ├── runtime_attr.cpp
│   ├── ir.cpp                  # pnnx Graph::load/parse、attribute 加载
│   ├── store_zip.cpp
│   ├── load_data.cpp
│   ├── tensor.cpp
│   ├── tensor_utils.cpp
│   ├── layer/
│   │   ├── abstract/
│   │   │   ├── layer.cpp
│   │   │   ├── layer_factory.cpp
│   │   │   └── param_layer.cpp
│   │   └── details/            # 具体算子
│   │       ├── relu.cpp/hpp
│   │       ├── sigmoid.cpp/hpp
│   │       ├── linear.cpp/hpp
│   │       ├── convolution.cpp/hpp
│   │       ├── maxpooling.cpp/hpp
│   │       ├── flatten.cpp/hpp
│   │       ├── softmax.cpp/hpp
│   │       ├── adaptive_avgpooling.cpp/hpp
│   │       ├── expression.cpp/hpp
│   │       ├── upsample, silu, cat, yolo_detect 等
│   └── parser/
│       └── parser_expression.cpp
│
├── test/
│   ├── test_ir/                # 图/算子/operand 解析测试
│   │   └── test_ir.cpp
│   ├── test_net/               # 端到端网络测试
│   │   ├── test_yolo.cpp       # test_network.yolov5
│   │   ├── test_resnet.cpp     # test_network.resnet
│   │   ├── image_util.cpp/hpp   # 预处理、Letterbox、NMS、ScaleCoords
│   ├── test_layer/             # 单层算子测试
│   │   ├── test_relu.cpp
│   │   ├── test_sigmoid.cpp
│   │   ├── test_conv.cpp
│   │   └── test_maxpooling.cpp
│   ├── test_topo/              # 拓扑排序与 Build 测试
│   │   └── test_topo.cpp
│   ├── test_tensor/            # 张量创建、reshape、transform 等
│   └── test_parser/            # 表达式解析测试
│
├── model_file/                 # 示例模型与图片
│   ├── *.pnnx.param / *.pnnx.bin
│   └── 31.jpg, car.jpg 等
│
└── build/                      # 构建目录（cmake 生成）
    └── FantasyInfer             # 可执行文件
```

## License

见项目根目录 `LICENSE` 文件。
