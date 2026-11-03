# 数据接口与准备

`data.py` 与 `prepare.py` 独立于旧项目包。运行环境需要 Python、PyTorch、NumPy、Pillow；Parquet 导出另需 `pyarrow`、`pycocotools`。不依赖本机 `D:`、旧 `curvi_*` 导入或网络连接。

## 加载契约

```python
from sam_path_v1.data import NativeInstanceDataset, collate_fn
data = NativeInstanceDataset("/data/export/manifest.json", split="train",
                             canvas_size=1036, seed=0, allow_condition=True,
                             data_root="/data/export")
```

清单格式沿用冻结导出：`records` 每项含 `id, file_name, split, group_id, rgb_sha256, shape=[H,W], image, masks`，image 是相对根目录的 upright PNG，masks 是含 `[N,H,W]` 二值数组 `masks` 的 NPZ。文件哈希 `image_sha256/masks_sha256` 与解码布尔标签哈希 `decoded_masks_sha256` 存在时也严格验证。显式 `data_root` 优先于清单中的根目录，便于上传/迁移。

`__getitem__` 返回 `image:float32[3,S,S]`、`valid:bool[1,S,S]`、`masks:float32[N,S,S]`、`id`、`meta`。原始 RGB 除以255；只在右侧/下侧补到canvas，补边RGB为0.5。1036保留1024²原像素并补12像素。超尺寸直接报错，不resize/crop；不自动纠正EXIF、不按GT挑朝向。所有原样本像素、实例顺序保持；padding和多实例重叠像素不参与监督，重叠处目标mask清零，清零后完全空的实例移除，索引记入meta。不截断实例数量。`original_size=[H,W]`、补边、原实例索引、忽略像素量进入meta。

`collate_fn` 返回 `images[B,3,S,S]`、`valid[B,1,S,S]`、`targets`列表，每项含`masks/id/meta`。train无随机增强；`seed/epoch`保留供训练器审计和未来扩展，同一索引在不同epoch数据相同。`val`请求映射到清单`validation`，清单本身不改名、不重新划分。坏图/hash/shape/EXIF问题会抛异常；评价器须记录失败，不能静默删除该图。

**`valid`包含GT重叠位置，只能用于loss/metrics，不能传入模型attention、路径选择或任何预测输入。** 模型所需`padding_valid`必须仅由`meta.original_size`与canvas构造：原始H×W内全部true，补边false，与GT内容无关。

初始化校验全清单的唯一ID/源名/导出路径、group与解码RGB哈希跨split泄漏以及counts。实际像素在读取时逐图验证。此校验不证明近重复、模拟运行或场景独立；`condition_holdout_pilot`必须显式允许且`independent_scene_split=false`。论文不能把该2300条件pilot称为独立场景验证。

## 已有PNG/NPZ导出

优先使用现成数据，`validate`不复制、不训练。以下占位路径换成实际上传路径；默认强制2000训练/300验证：

```bash
python -m sam_path_v1.prepare validate --manifest /data/export/manifest.json --frozen-manifest sam_path_v1/protocols/pilot2300_manifest.json --data-root /data/export --max-instances 24 --allow-condition-pilot --receipt /data/validation_new.json
```

receipt必须是新文件；其父目录应已存在。`--frozen-manifest`会核对全部成员、ID、split、group、shape和RGB hash；只有当前清单时只能确认内部一致，receipt会注明未做冻结成员核对。推荐正式训练始终传冻结清单。

`--max-instances`默认24，必须与模型query容量一致。预检统计每图去掉重叠歧义像素后仍非空的可见实例数；超过容量直接报错，绝不截断标签。成功收据报告`max_visible_instances`（过滤后最大值）、`source_max`（原始NPZ实例最大值）和设置的`max_instances`。完全空或仅含重叠歧义像素的实例不占用可见query数；例如源30个实例过滤后2个，只需要容量2。Dataset本身不施加容量上限，保留完整可见监督，由预检和模型/损失层拒绝不适配的配置。

包内`protocols/pilot2300_manifest.json`是现有冻结清单的**成员比对收据**，全部2300条records及顺序保持原样，`data_root='.'`。它标明`membership_only=true`、`contains_decoded_data=false`并记录原清单SHA256；不含PNG/NPZ，不能把它当作数据已经上传的凭据。作为上面`--frozen-manifest`参数可在服务器直接使用，无需原电脑路径。

## 上传官方Parquet时

```bash
python -m sam_path_v1.prepare export-parquet --frozen-manifest sam_path_v1/protocols/pilot2300_manifest.json --parquet-root /data/WireSeg-36K --output /data/decoded_NEW --max-instances 24 --allow-condition-pilot
```

只依据冻结`file_name`提取全部2300成员，不按分数筛图、不重新划分；逐Arrow batch扫描，默认16行，最终清单保持冻结顺序。扫描所有本地shards以检测选中源名重复；未命中的RGB不解码。必须上传涵盖所有选中成员的shards；缺成员、重复成员、RGB hash/shape变化、非upright EXIF立即报错。仅支持本地数据，不下载。

导出写原生PNG与保留原实例顺序/重叠的布尔NPZ，逐项重读校验，再执行包含`--max-instances`的完整验证，全部通过后才写`completion.json`。容量超限仍保留原完整标签并报告失败，不能把没有completion的部分导出当作已完成。新目录限定防覆盖，失败会保留部分输出和`failure.json`，不自动续写。空间需求取决于源图/标签，执行前由部署者确认可用磁盘。冻结旧清单若没有标签hash，则本工具不能证明新上传RLE与历史标签逐像素相同；导出会固定本次mask字节/解码hash供后续复核，不能将RGB hash验证夸大为历史标签hash验证。

小型软件fixture可显式`--allow-small-fixture`放开2000/300计数，收据标`explicit_fixture`；这不代表正式实验数据。CPU测试入口：

```bash
python -m unittest discover -s sam_path_v1/tests -p test_data.py -v
```

测试覆盖原像素/灰色padding、重叠ignore、可变/零实例、路径安全、split泄漏、EXIF/hash失败、冻结成员、流式导出顺序/缺失/重复。`pyarrow`缺失时只跳过真实微型Parquet集成项，其余不依赖Parquet运行；不扫描2300张数据。

## 2026-10-28 本轮已验证

17项CPU测试全部通过，包含微型真实Parquet batch=1导出、压缩与非压缩COCO RLE roundtrip，以及过滤后容量检查、超限导出失败、包内冻结收据检查。测试进程临时加载本机已有`third_party`依赖，生产包没有引用旧包或固定依赖路径。另从现成完整2300导出的train/validation各读一图，确认清单2000/300、原图1024²、batch为`[2,3,1036,1036]`、目标实例分别16/6、补边0.5且valid=false。只读取两张真实导出样本；没有全量预处理、复制大数据或启动训练。这是接口/格式验证，不是实验精度结果。包内冻结收据的全部2300条records与原清单逐对象精确相等，包含顺序；该核验只读取两份JSON。
