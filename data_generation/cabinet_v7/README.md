# 控制柜线束 RGB-D 数据生成 v7

三档分别保留 1、2、3 套线束，每档 4 个共同机位，原生分辨率 1024×1024。三档共用控制柜、灯光和已求解的线束几何。柜内包含 DIN 导轨、端子排、开槽线槽和模块，线夹采用带缓冲内衬和实体紧固件的圆形卡箍。

## 生成

安装 Blender 4.2 和 Python 3.10+，准备至少 1 GiB 可用空间。从仓库根目录运行：

```bash
python -m pip install -r data_generation/cabinet_v7/requirements.txt
python data_generation/cabinet_v7/create_dataset.py --blender blender --output outputs/cabinet_v7
```

`--blender` 可填可执行文件的完整路径。`--device GPU` 使用支持 CUDA 的 NVIDIA 显卡，默认使用 CPU。`--output` 指向新目录，保证重复运行保留原输出。日志在输出目录的 `logs/`。

生成过程先用 `seed/` 中的控制点、冻结力学解和相机参数重建基础场景，再替换卡箍、搭建控制柜并渲染。默认流程直接使用已校验的轴线。原始静态杆求解器与参数保存在 `seed/solve_rods.py` 和 `seed/solver_config.json`，复跑求解器额外需要 SciPy 和 PyTorch。

```bash
python data_generation/cabinet_v7/seed/solve_rods.py \
    --input data_generation/cabinet_v7/seed/input_contract.json \
    --rest-ratio 1.01 --maxiter 6000 --output outputs/rod_solution
```

## 输出

- `delivery/panel_harness_rgbd_v7.zip` 包含 RGB PNG、RGB-D NPZ、米制深度真值/模拟输入、整数 ID 掩码、相机参数、中心线与拓扑。解压后按 `manifest.jsonl` 读取，包内附加载示例。
- `delivery/final/index.html` 展示 12 帧及各自深度、整束、分支和可见单线标签。
- `data/` 保存 Blender 场景及未打包的渲染通道，`receipts/` 保存几何、深度及标签检查结果和归档哈希。

卡箍为 `clip_tie` 辅助部件，主目标是线束的可见表面。原始线束 ID 跨场景保留。三档属于同一合成场景家族，划分字段为 `quality_pilot_only`。相机和深度噪声属于仿真设置。Blender 版本和渲染设备会影响像素值及生成文件哈希。

## 来源

这套柜体、卡箍和线束种子由本项目生成。种子文件和原始 v7 版本摘要记录在 [seed/provenance.json](seed/provenance.json)。卡箍和导轨安装方式分别参考 [NORMA RSGU](https://www.norma-connects.com/en/products/norma-rsgu-rubber-lined-pipe-clips-DIN-3016) 与 [Phoenix Contact 安装示意](https://assets.phoenixcontact.com/file/fcda5126-466b-41d4-827d-91d1a2e9bfad/media/original?1082999_EN_HQ_HOUSING_Basics_LoRes.pdf=)。仓库保存项目源码和小型合成种子，发布文件由上述命令生成。
