# 文件及目录说明
该目录下存储了各种文件和目录
- flake.nix：定义了该项目所需的全部环境，当然也可以通过如requirements.txt方式手动安装
- flake.lock：版本锁文件，类似于cargo.lock
- 其他*.nix：其他软件包和环境定义
- 其他文件及文件夹：命名规则、说明见子文件夹的README.md
# 实验设置
- 系统环境：
  - CPU: AMD Ryzen 5 7500F
  - GPU: AMD Radeon RX 6700 XT
  - OS: NixOS 25.05 (Warbler) x86_64
  - Kernel: Linux 6.13.9-xanmod1
- 运行环境：已在flake.nix中定义
# 项目亮点
- 部分训练数据来自校园，训练的模型可应用于检测校园树木健康状况
- pytorch使用了rocm后端（YOLO很可惜不支持vulkan后端），不使用cuda，对a卡友好
# 运行方式
1. 安装nix,nix可在WSL2上使用
2. nix run .#python312FHSEnv
3. python ./code/inference.py (由于模型文件被.gitignore忽略，此命令仅展示用)