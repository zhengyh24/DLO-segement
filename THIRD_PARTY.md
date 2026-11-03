# 第三方来源

| 组件 | 来源 / 版本 | 许可与使用方式 |
|---|---|---|
| SAM3.1 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3)，`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` | 按上游代码与模型条款单独安装、获取权重 |
| HQ-SAM | [SysCV/sam-hq](https://github.com/SysCV/sam-hq)，Python 包 `segment-anything-hq==0.3` | 单独安装；SACM 分支中的编码器文件附原许可证 |
| GeoMPrompt | [项目源](https://gitlab.geomdata.com/rootshape/geomprompt)，`0.2.5` | AGPLv3，使用已发布 Python 包 |
| mBEST | [StructuresComp/mBEST](https://github.com/StructuresComp/mBEST)，`b8c19b217185449838b0b55f406c4577a5394ba0` | 源码随仓库提供；[GPL-3.0](third_party/mBEST/LICENSE) |
| WireSeg-36K | [DeformX/WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K) | CC BY 4.0；数据集由原作者提供，见 [原始数据卡](data/WireSeg-UPSTREAM.md) |

mBEST 保留已有 `np.int → int` 兼容修改；`build.py` 是可移植编译入口。项目包装器扩展显示颜色数量，并将输出限制到可见前景。第三方代码与数据各自保留其上游许可。

SAM + tracker 使用 [SAM2.1 Hiera-Tiny](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4) 做第二骨干，另行获取官方源码与基础权重。Tiny 权重的 SHA-256 为 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。SAM3/SAM2 的实际基础权重及引擎 SHA 均记录于各训练目录的 model_provenance.json。

本分支的 PADLOS 对照包含调用方与评价数值。上游实现和权重按 [实验协议](results/sam_tracker/padlos_test1800/protocol.json) 的提交及哈希单独获取。HANDLOOM 只发布 [数据审计](results/sam_tracker/handloom/) 与描述性统计，来源固定为 [官方提交](https://github.com/vainaviv/handloom/tree/4b23ef54ccb85d814965f8ae1c22e91f3e5171ed)。WireSeg 派生的合成图示、样本成员和标注哈希来自 [DeformX/WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K)，按上游 CC BY 4.0 注明来源。
