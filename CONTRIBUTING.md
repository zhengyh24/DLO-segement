# 协作与提交规范

## 分支

- `main`：SAM3.1 / coverage_color 基线。
- `sacm`：可训练前景模型。
- `curvi-tracker`：Curvi 思路下的身份学习和直接实例原型。
- 新功能从对应分支创建 `feat/<简短主题>`，修复使用 `fix/<简短主题>`；完成后提交 PR 到对应路线。
- 新实验使用独立配置和结果目录。已开始运行的源码快照、数据划分和选择规则保持固定。

## Commit 格式

```text
<type>(<scope>): <一句话说明改动>
```

`type` 使用 `feat`、`fix`、`refactor`、`test`、`docs`、`data`、`chore`；`scope` 使用 `sam31`、`sacm`、`curvi`、`eval`、`dataset` 或具体模块名。中文和英文均可，每次提交只解决一个主题。

```text
feat(curvi): 增加实例状态更新模块
fix(eval): 修正边缘补齐区域的计分
data(dataset): 添加同色交叉样本与来源清单
docs(sam31): 补充权重配置步骤
```

需要解释时在标题后空一行，写明改动原因、行为变化和验证结果。

## 提交内容

- 代码连同配置、必要检查及简短使用说明一起提交。
- 新数据附来源、许可、原始文件名、划分与 SHA-256；保留原生坐标。
- 实验摘要注明数据版本、种子、成功更新数、权重选择方法、IoU/PQ、失败数量和成本。
- 大型权重放 GitHub Release；数据按有用子集提交，完整数据提供获取和导出方法。
- 不提交运行环境、缓存、密钥、机器账户信息或成批中间输出。

## 提交前

```bash
python scripts/verify_repository.py
git diff --check
git status --short
```

再运行与改动相关的检查。PR 描述用三项即可：**改了什么、为什么改、如何验证**。比较模型时使用同一划分和指标，训练中断与负结果也保留在结果摘要里。
