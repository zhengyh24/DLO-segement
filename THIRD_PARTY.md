# 第三方来源

| 组件 | 来源 / 版本 | 许可与使用方式 |
|---|---|---|
| SAM3.1 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3)，`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` | 按上游代码与模型条款单独安装、获取权重 |
| HQ-SAM | [SysCV/sam-hq](https://github.com/SysCV/sam-hq)，Python 包 `segment-anything-hq==0.3` | 单独安装；SACM 分支中的编码器文件附原许可证 |
| GeoMPrompt | [项目源](https://gitlab.geomdata.com/rootshape/geomprompt)，`0.2.5` | AGPLv3，使用已发布 Python 包 |
| mBEST | [StructuresComp/mBEST](https://github.com/StructuresComp/mBEST)，`b8c19b217185449838b0b55f406c4577a5394ba0` | 源码随仓库提供；[GPL-3.0](third_party/mBEST/LICENSE) |
| WireSeg-36K | [DeformX/WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K) | CC BY 4.0；数据集由原作者提供，见 [原始数据卡](data/WireSeg-UPSTREAM.md) |

mBEST 保留已有 `np.int → int` 兼容修改；`build.py` 是可移植编译入口。项目包装器扩展显示颜色数量，并将输出限制到可见前景。第三方代码与数据各自保留其上游许可。
