import torch
import torch.nn as nn
from torchvision.models import densenet121
import torch_pruning as tp


# 定义重要性计算函数
class MyMagnitudeImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = []
        # 2. 迭代分组内的各个层，对Conv层计算重要性
        for dep, idxs in group:  # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.target.module  # 获取 nn.Module
            prune_fn = dep.handler  # 获取 剪枝函数
            # 3. 这里我们简化问题，仅计算卷积输出通道的重要性
            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                w = layer.weight.data[idxs].flatten(1)  # 用索引列表获取耦合通道对应的参数，并展开成2维
                local_norm = w.abs().sum(1)  # 计算每个通道参数子矩阵的 L1 Norm
                group_imp.append(local_norm)  # 将其保存在列表中

        if 0 == len(group_imp): return None  # 跳过不包含卷积层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp


# 定义模型和输入
model = densenet121(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)

# 计算重要性
imp = MyMagnitudeImportance()
# 忽略无需剪枝的层，例如最后的分类层
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

# 初始化剪枝器
# 定义重复5次剪枝
iterative_steps = 5
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5,  # 目标移除50%的通道
    ignored_layers=ignored_layers,
)

# pruning-finetuning循环
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()  # 每次剪枝10%
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
    print("  Iter %d/%d, MACs: %.2f G => %.2f G" % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9))

print(model)
