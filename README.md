## 环境配置命令

创建并激活Python 3.10虚拟环境

`conda create -n yolov8_env python=3.10 -y`

`conda activate yolov8_env`

安装CUDA 13.0兼容的PyTorch Nightly版本

`pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130`

安装YOLOv8官方库

`pip install ultralytics`

安装Streamlit

`pip install streamlit`
