<!--
SPDX-License-Identifier: Apache-2.0
DeepSpeed Team
-->

# Qwen3-VL-MoE 在 MUSA 上的 DeepSpeed AutoEP 适配指南

本文面向第一次接触 MoE、专家并行和 ZeRO-3 的开发者，说明本分支如何让
Qwen3-VL-MoE 在 MUSA 设备上使用 DeepSpeed AutoEP，以及为什么这项工作并不只是增加一段 JSON 配置。

本文覆盖以下内容：

- AutoEP、专家并行、ZeRO-3 和 DeepEP 分别是什么；
- Qwen3-VL-MoE 为什么需要专门适配；
- 模型从 Hugging Face MoE 层转换为 AutoEP 层的完整过程；
- MUSA Transformer Engine（TE）GroupedGEMM 后端的前向和反向数据流；
- MUSA 2.7.x 的稳定 top-k 和专家计数正确性修复；
- AutoEP 替换 Parameter 后如何重绑定已提前创建的 client optimizer；
- AutoEP、ZeRO-3 和 MCCL 多进程组通信为什么可能卡住，以及保守串行模式如何工作；
- ZeRO-3 checkpoint 为什么要记录后端相关的专家参数名；
- 推荐配置、验证方法、已知限制和排障步骤。

> 本文记录的是当前分支在特定软件栈上的工程适配，不代表所有 DeepSpeed、PyTorch、
> torch_musa、Transformer Engine 或 MCCL 版本都具有相同行为。升级任一组件后都应重新执行数值、
> 性能、显存和分布式稳定性验证。

## 1. 先理解几个基本概念

### 1.1 什么是 MoE

普通稠密 MLP 会让每个 token 经过同一组 FFN 参数。Mixture of Experts（MoE）把一个 MLP
替换成多个“专家”，再由 Router 为每个 token 选择少量专家。

Qwen3-VL-30B-A3B 的文本骨干包含 128 个专家，每个 token 选择 8 个专家：

```text
token hidden state
    -> Router 计算 128 个专家分数
    -> 选择 top-8 专家
    -> token 分发到 8 个专家
    -> 专家 FFN 计算
    -> 按 Router 权重加权合并
```

模型拥有很多专家参数，但一次只激活其中一部分，因此可以在扩大参数量的同时控制计算量。

### 1.2 什么是专家并行（EP）

如果每张卡都保存全部 128 个专家，显存开销会很大。专家并行把专家分散到多张卡上。

当前配置为 `autoep_size=8`：

```text
128 个专家 / 8 个 EP rank = 每个 EP rank 保存 16 个专家
```

当本地 token 选择了其他 EP rank 上的专家时，需要通过 All-to-All 把 token 发送到专家所在 rank，
专家计算完成后再通过一次 All-to-All 把结果送回来源 rank。

在 32 卡、EP size 8 的拓扑中，可以直观理解为：

- 4 个 EP group，每组 8 个 rank；
- 同一个 EP 位置在 4 个副本之间组成 expert-data-parallel（EDP）group；
- Router、Attention 等非专家参数仍按普通数据并行/ZeRO 规则处理；
- 专家参数按其 EDP group 做梯度归约和 ZeRO 分片。

### 1.3 什么是 AutoEP

AutoEP 是 DeepSpeed 的自动专家并行模块。用户仍然加载原始 Hugging Face 模型，
DeepSpeed 在 `deepspeed.initialize()` 阶段自动完成：

1. 根据模型 preset 找到所有 MoE 层；
2. 识别 Router 和专家权重；
3. 创建 EP/EDP 进程组；
4. 把原始 MoE 层替换成 `AutoEPMoELayer`；
5. 将专家权重切分到对应 EP rank；
6. 接入 token reorder、All-to-All 和 GroupedGEMM；
7. 让 ZeRO、optimizer 和 checkpoint 识别专家参数的特殊归属。

它的价值是：模型源码不需要手工改写成 DeepSpeed MoE 层。

### 1.4 AutoEP 和 DeepEP 不是一回事

二者名字相似，但职责不同：

| 模块 | 主要职责 |
| --- | --- |
| AutoEP | 识别和替换模型 MoE 层、创建并行组、管理专家参数和训练语义 |
| DeepEP | 提供面向 MoE dispatch/combine 的高性能通信实现 |

当前适配使用 AutoEP 和 DeepSpeed/MCCL All-to-All，没有接入 DeepEP。未来可以让 AutoEP 的
dispatch/combine 调用 DeepEP，但那是独立的重大通信后端改造，不能把二者当作一个开关。

### 1.5 ZeRO-3 为什么让问题更复杂

ZeRO-3 不仅分片 optimizer 状态和梯度，还分片模型参数。参数在真正计算前通过 AllGather 获取，
计算后又可能释放。

AutoEP 加入后，同一个进程里至少出现以下通信域：

- 全局数据并行/普通参数的通信组；
- EP group 的 token All-to-All；
- EDP group 的专家参数和专家梯度通信；
- ZeRO-3 参数 AllGather 和梯度 ReduceScatter。

如果不同 rank 以不同顺序在不同 stream 上进入这些 collective，就可能形成循环等待。因此，
“AutoEP 本身已有”并不代表它天然适配当前 Qwen3-VL、MUSA TE、MCCL 和 ZeRO-3 组合。

## 2. 本次适配解决了什么问题

### 2.1 Qwen3-VL 的 MoE 配置位于 `text_config`

Qwen3-VL 是多模态模型，顶层 config 同时包含视觉和文本配置。`num_experts`、
`num_experts_per_tok` 等路由属性位于 `model.config.text_config`，而不是顶层 config。

如果沿用普通文本 MoE preset，AutoEP 可能找不到正确的专家数和 top-k。新增的
`qwen3_vl_moe` preset 会：

- 只匹配 `model.language_model.layers.<N>.mlp`；
- 从 `text_config` 读取路由属性；
- 识别 `gate` Router；
- 识别 `experts.gate_up_proj` 和 `experts.down_proj`；
- 要求 Transformers 5.2.0 或更新版本。

这样不会误把视觉塔中的普通 MLP 当成文本 MoE 层。

### 2.2 Qwen3-VL 使用融合专家权重

通用 AutoEP expert 后端使用三个参数：

```text
w1 = gate projection
w3 = up projection
w2 = down projection
```

Qwen3-VL 的 Hugging Face 实现把 gate 和 up 合并成一个三维参数：

```text
gate_up_proj: [E, 2 * ffn_hidden, hidden]
down_proj:    [E, hidden, ffn_hidden]
```

本模型中：

```text
E = 128
hidden = 2048
ffn_hidden = 768
```

当 EP size 为 8 时，每个 rank 的 MUSA TE 参数形状为：

```text
gate_up_proj: [16, 1536, 2048]
down_proj:    [16, 2048, 768]
```

MUSA TE 后端保留这个融合布局，避免先拆成 `w1/w3` 再在计算前重新组织。

### 2.3 MUSA 2.7.x 的路由正确性问题

当前环境中确认了两类问题：

1. 分数相等时，原生 top-k 的 expert 选择顺序不稳定；
2. MUSA 上的 `torch.bincount` 在生产规模、偏斜路由分布下可能静默漏计。

专家计数不是普通统计信息。它决定 token reorder、All-to-All split 和每个专家的 GroupedGEMM
行数。少计 token 会造成错误的 dispatch 元数据，甚至出现“训练特别快但数值不对”的假象。

### 2.4 MUSA/MCCL 多 stream、多进程组通信可能互相等待

ZeRO 参数获取、ZeRO 梯度归约和 AutoEP token All-to-All 原本可以使用各自的 stream。
但在当前 MCCL 组合中，多进程组 collective 并发提交曾出现 hang。

本次没有强制把所有工作放到同一个 stream，而是新增保守模式：

```text
保留各子系统选择的 stream
    +
同一进程内的 DeepSpeed collective 严格串行完成
```

这解决的是稳定性和 collective 顺序问题，代价是牺牲通信异步重叠。

### 2.5 ZeRO-3 checkpoint 不能再假设专家参数名固定为 `w1/w2/w3`

MUSA TE 后端的直接参数名是 `gate_up_proj/down_proj`。如果 checkpoint 代码仍硬编码
`w1/w2/w3`，保存、加载和 Universal Checkpoint 转换都会遗漏或错误重建专家权重。

因此 checkpoint 元数据新增了每层真实的 `expert_parameter_names`，同时对旧 checkpoint
保留 `w1/w2/w3` 兼容路径。

### 2.6 AutoEP 替换层后必须同步更新 optimizer 参数对象

LLaMAFactory 等上层框架可能在调用 `deepspeed.initialize()` 之前就创建 client optimizer。
optimizer 的 `param_groups` 此时保存的是原始 Hugging Face Router 和专家 `Parameter` 对象。
AutoEP 随后会创建新的 `AutoEPMoELayer`，并为 Router 和本地专家创建新的 `Parameter` 对象；
复制权重数值不会保留 Python 对象身份。

如果只替换模型而不更新 optimizer，ZeRO-3 会从旧 `param_groups` 建立分片和梯度 hook：

```text
forward/backward 使用新的 AutoEP 参数
    !=
optimizer/ZeRO-3 持有的旧 MoE 参数
```

这种错误不会必然 OOM 或 hang。训练甚至会异常快，但新的 Router/专家参数不会进入 optimizer 更新，
global grad norm 也会主要反映非专家参数，最终表现为 loss 或收敛速度偏离。

当前修复让 AutoEP 在替换时返回精确的参数映射计划，并在创建 ZeRO optimizer 之前：

1. 按本地 expert slice 建立旧参数到新参数的映射；
2. 从 `model_parameters` 和 client optimizer 中移除非本 EP rank 的远端专家；
3. 将本地源 Router/专家替换为新的 AutoEP `Parameter`；
4. 保留原 optimizer 对象、param-group 顺序、学习率和 weight decay 等组属性；
5. 校验 optimizer 参数集合与当前模型 trainable 参数集合一致。

映射禁止按名称或 shape 猜测。遇到部分源参数、跨 param-group 合并、重复参数、未映射旧参数，
或源参数已经拥有 optimizer state、梯度和 backward hook 时会在 ZeRO 初始化前直接报错，避免静默少训参数。

## 3. 从初始化到一次训练 step 的完整流程

### 3.1 初始化流程

```text
读取 DeepSpeed JSON
    -> 解析 expert_parallel
    -> 校验 enabled、autoep_size、backend 和拓扑
    -> 从 registry 取得 qwen3_vl_moe preset
    -> 从 model.config.text_config 读取 E=128、top_k=8
    -> 只检测 48 个文本 MoE MLP
    -> 创建 EP/EDP group
    -> 在 ZeRO GatheredParameters 作用域中临时还原源专家参数
    -> 每个 EP rank 切出自己的 16 个专家
    -> 创建 AutoEPMoELayer + MusaTEGroupedExperts
    -> 复制 Router 和本地专家权重
    -> 生成原 Parameter 到新 Parameter 的精确映射计划
    -> 重绑定 model_parameters 和已提前创建的 client optimizer
    -> 校验 param-group 语义和当前模型参数身份
    -> 标记专家参数的并行归属
    -> 进入 ZeRO-3 optimizer 初始化
```

源参数必须先 gather，是因为 Hugging Face 模型可能已经在 ZeRO Init 上下文中构造；此时
`gate_up_proj/down_proj` 在单个 rank 上只是分片，不能直接按专家维度切片。

### 3.2 前向流程

设当前 rank 有 `T` 个 token，每个 token 选择 `k=8` 个专家，则路由 assignment 总数约为
`M=T*k`。

```text
hidden[T, 2048]
    -> Router softmax
    -> stable top-k，得到 expert_ids[T, 8]
    -> 正确统计每个专家的 token 数
    -> TokenReorderer 按目标 EP rank/专家排序
    -> 第一次 All-to-All：把 token 发往专家所在 rank
    -> 根据本地 16 个 expert counts 形成 m_splits
    -> TE GroupedGEMM FC1: 2048 -> 1536
    -> SwiGLU: 1536 -> 768
    -> TE GroupedGEMM FC2: 768 -> 2048
    -> 第二次 All-to-All：把专家输出送回来源 rank
    -> restore order
    -> 乘 route weights 并对 top-8 结果求和
    -> output[T, 2048]
```

### 3.3 反向流程

自定义 autograd 会分别计算：

- FC2 对输入的梯度；
- FC2 权重梯度；
- SwiGLU 反向；
- FC1 对输入的梯度；
- FC1 权重梯度；
- token dispatch/combine 的反向 All-to-All；
- Router 和 route weight 梯度。

MUSA TE GroupedGEMM 使用的主要 layout 为：

| 方向 | 作用 | TE layout |
| --- | --- | --- |
| forward | `X @ W^T` | 默认路径 |
| dX | `dY @ W` | `NN` |
| dW | `dY^T @ X` 的等价分组形式 | `NT` |

零 token expert 和尾部 dispatch padding 会显式返回正确形状的零值，避免空 GroupedGEMM
或未初始化输出污染训练。

## 4. 配置方法

### 4.1 最小 AutoEP 配置

```json
{
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 8,
    "preset_model": "qwen3_vl_moe",
    "expert_backend": "musa_te",
    "serialize_communications": true
  }
}
```

字段说明：

| 字段 | 含义 | 当前建议 |
| --- | --- | --- |
| `enabled` | 是否启用 AutoEP 替换 | 正式 AutoEP 训练设为 `true` |
| `autoep_size` | 一个 EP group 中的 rank 数 | 当前模型使用 8；必须整除 128 |
| `preset_model` | 模型结构识别规则 | Qwen3-VL 使用 `qwen3_vl_moe` |
| `expert_backend` | 专家计算后端 | MUSA TE 路径必须显式设为 `musa_te` |
| `use_grouped_mm` | 通用 `auto` 后端是否使用 `torch._grouped_mm` | `musa_te` 不依赖该字段 |
| `serialize_communications` | 是否启用保守 collective 串行 | 当前 MUSA/MCCL 组合建议 `true` |

`expert_backend="auto"` 在当前实现中表示保留原有通用 AutoEP 行为，并不会自动选择
`musa_te`。如果期望使用 MUSA TE，必须显式配置。

### 4.2 当前项目验证配置

下面是当前项目使用的核心组合。ZeRO bucket 和 live/reuse 数值是特定模型、显存和 Trace
实验的结果，不应直接复制到其他模型或硬件：

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "allgather_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 3774873,
    "stage3_param_persistence_threshold": 21233664,
    "stage3_max_live_parameters": 2e10,
    "stage3_max_reuse_distance": 4e10,
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 8,
    "preset_model": "qwen3_vl_moe",
    "expert_backend": "musa_te",
    "use_grouped_mm": false,
    "serialize_communications": true
  }
}
```

注意：`overlap_comm=true` 与 `serialize_communications=true` 并不矛盾。前者仍让 ZeRO
建立并使用原来的通信 stream；后者会等待每个 DeepSpeed collective 完成后才放行下一个
collective。因此当前安全模式保留 stream 归属，但不宣称存在 collective 之间的异步重叠。

### 4.3 拓扑约束

- `num_experts % autoep_size == 0`；
- world size 必须与 EP/DP 拓扑兼容；
- 当前 ZeRO-3 AutoEP 不支持同时启用 AutoTP；
- 不支持来自 `mpu` 的 tensor model parallel；
- 不支持 sequence parallel、MiCS、hpZeRO secondary group、非 1 的 expert tensor parallel；
- 不支持 quantized gradients；
- 当前验证使用 BF16、micro-batch 1；
- 当前正式训练使用 32 卡和 GA=8，global batch size 为 `32 * 1 * 8 = 256`。

功能冒烟测试可以先使用 `autoep_size=1`。这会执行 AutoEP 层替换，但所有专家仍在本地，
不会产生 EP All-to-All，因此不能替代多机通信验证。

## 5. MUSA TE GroupedGEMM 后端

### 5.1 为什么单独增加后端

通用 `GroupedExperts` 面向 `w1/w2/w3`。MUSA TE 使用融合 `gate_up_proj`，并通过
`general_grouped_gemm` 根据每个专家的 token 数执行 ragged-M GroupedGEMM。

新增 `MusaTEGroupedExperts` 的收益是：

- 保留 Qwen3-VL 原始融合权重布局；
- FC1、FC2 的 forward、dX 和 dW 都走 TE GroupedGEMM；
- 激活优先调用平台 `swish_glu`，否则回退为 `silu(gate) * up`；
- expert 参数继续是原生 `nn.Parameter`，可被 ZeRO 和 optimizer 管理。

### 5.2 运行前置条件

`expert_backend="musa_te"` 要求：

- 安装 `torch_musa`；
- 安装与 MUSA 兼容的 Transformer Engine；
- 输入和权重都位于 MUSA device；
- 输入/权重 dtype 一致，且为 FP16 或 BF16；
- 输入为二维，权重为三维；
- 每个专家必须有一个 token count。

该后端是显式选择，缺少依赖或输入不满足条件时直接报错，不会静默切回另一套专家实现。
这样可以避免用户以为自己正在测 TE，实际上却落入慢速 fallback。

### 5.3 首次调用编译

MUSA TE kernel 可能在第一次真实调用时编译。首步还可能包含 communicator 初始化、allocator
扩容、autotune 和缓存填充。因此：

- 不使用 step1 判断稳态性能；
- 至少预热 2 步；
- 比较 step3 之后的多步 P50/P95；
- A/B 两边都必须使用已经完成编译的相同软件环境。

### 5.4 当前仍存在的 host 同步

TE API 的 `m_splits` 当前需要 Python `list[int]`。实现会把本地 expert counts 复制到 CPU：

```text
tokens_per_expert.detach().to("cpu").tolist()
```

因此当前适配没有实现完全 device-resident 的 `m_splits`。这会产生同步，但在现有 TE API
约束下不能仅靠修改 AutoEP Python 层消除。未来需要 TE/MATE 支持 device offsets/counts。

## 6. 路由正确性修复

### 6.1 稳定 top-k

在 MUSA + torch 2.7.x 上，AutoEP Router 对完整专家分数执行稳定降序排序，再取前 k 个：

```text
stable sort(descending=True) -> [:top_k]
```

当多个 expert 分数相等时，稳定排序保留原 expert ID 顺序。例如所有分数相同时，top-2
固定选择 expert 0 和 1，而不是随 kernel 行为变化。

该兼容路径只对 device type 为 `musa` 且 torch 版本为 2.7.x 生效；CPU、CUDA 和后续
torch 版本继续使用原 `torch.topk` 快路径。

稳定全排序的复杂度高于只取 top-k，所以它首先是正确性修复，不应直接宣称性能收益。

### 6.2 可靠的 `ep_count`

真实生产 shape 验证覆盖：

- 9 种 token 长度；
- top-k 8、128 个专家；
- 10 个随机种子；
- 3 类路由分布；
- 总计 270 个 case。

结果：

- 原生 MUSA `torch.bincount` 有 16/270 个 case 与 CPU 结果不一致；
- 错误都出现在偏斜路由分布；
- 单个 case 最多漏计 144 个 assignment；
- 修复后的可信计数路径为 0/270 不一致。

因此 MUSA/torch 2.7.x 会将选中的 expert IDs 复制到 CPU 做 `bincount`，再把 counts
送回原 device，并校验：

```text
counts 长度不超过 num_experts
sum(counts) == expert_ids.numel()
```

这个 workaround 会引入 device-to-host 同步。后续如果平台提供经过生产 shape 验证的
device-side count kernel，可以在保留完整性校验的前提下替换；在此之前不能为了速度恢复
已知会漏 token 的原生路径。

### 6.3 为什么 loss 正常还不够

路由错误可能只影响部分 token，单步 loss 看起来仍然“接近”。因此最低验证必须同时包含：

- expert IDs；
- 每个 expert count；
- count 总和；
- dispatch/restore 覆盖；
- forward 输出；
- hidden、Router 和 expert 权重梯度；
- 多步 loss 和 global grad norm。

## 7. 通信串行模式

### 7.1 原问题

不同子系统可能按以下顺序提交通信：

```text
rank A: world AllGather -> EP AllToAll -> EDP ReduceScatter
rank B: EDP AllGather   -> world collective -> EP AllToAll
```

如果 MCCL 后端允许这些工作在多个内部 stream/进程组中同时未完成，rank 之间的等待关系可能
形成环路。只在 Python 侧设置 `async_op=False` 或只记录 caller stream event，也不一定覆盖
MCCL 内部 stream 上已经排队的 work。

### 7.2 当前解决方式

启用 `serialize_communications` 后：

1. 使用进程内 lock 确保一个时刻只提交一段 DeepSpeed 通信；
2. 嵌套调用通过 thread-local depth 合并，避免同一线程重复加锁；
3. collective 返回 Work-like 对象时显式调用 `wait()`；
4. 在 caller stream 记录并同步 event；
5. 当前 collective 完成后才进入下一个通信 epoch。

该逻辑覆盖：

- AutoEP split count 交换；
- token dispatch/combine All-to-All 及其反向；
- ZeRO-3 参数 AllGather；
- ZeRO-3 梯度 ReduceScatter；
- 通过 DeepSpeed communication facade 发起的其他 collective，包括 optimizer 尾部的
  overflow/grad-norm 通信。

### 7.3 ZeRO 参数获取还需要按 process group 拆分

ZeRO 的一次预取 bucket 可能同时看到普通参数和专家参数。二者的 `ds_process_group` 不同，
不能放入同一个 coalesced AllGather。

当前实现先按 process group 分组，再使用确定顺序提交：

1. 普通 replicated/global 参数组；
2. AutoEP expert 参数组；
3. 同类组内按最小 `ds_id` 保持稳定顺序。

这既避免把不同 process group 的参数错误合并，也减少各 rank collective 顺序不一致的风险。

### 7.4 能否多 stream 但通信串行

可以，这正是当前模式：ZeRO AllGather stream、梯度归约 stream 和 AutoEP 所在 stream 仍然存在，
但 collective 的生命周期不相互重叠。

该模式的优点是容易证明顺序和 buffer 生命周期；缺点是会损失通信隐藏机会。只有在 Trace
证明后端和 process-group 顺序安全，并经过长时间多机验证后，才应把
`serialize_communications` 设为 `false`。

## 8. ZeRO-3 参数和 checkpoint 适配

### 8.1 专家参数如何标记

MUSA TE 后端把以下参数标记为 expert group：

```text
experts.gate_up_proj
experts.down_proj
```

它们按 EDP group 做专家参数处理，而 Router 和非专家参数使用普通数据并行/ZeRO 组。

### 8.2 checkpoint 元数据

每个 AutoEP 层的 metadata 记录：

- module path 和 layer ID；
- 全局/本地 expert 数；
- EP size、EP rank、EDP rank/world size；
- 当前 rank 对应的全局 expert 范围；
- expert key prefix；
- `expert_parameter_names`；
- ZeRO-3 分片格式和版本。

通用后端的参数名通常是 `w1/w2/w3`，MUSA TE 后端则是
`gate_up_proj/down_proj`。读取旧 metadata 时，如果没有 `expert_parameter_names`，代码仍按
`w1/w2/w3` 解释，保持向后兼容。

### 8.3 保存、加载和 Universal Checkpoint

checkpoint 代码不再硬编码参数名，而是读取每层的真实 direct parameter names，用于：

- 从 per-expert 文件重新 stack 本地专家参数；
- 过滤 non-MoE state dict；
- 生成专家参数匹配 pattern；
- 重建 optimizer state；
- 转换 Universal Checkpoint。

普通 AutoEP ZeRO-3 checkpoint 适合相同拓扑恢复。需要改变数据并行 world size 或
`autoep_size` 时，应使用 Universal Checkpoint 转换流程，并确认新的 `autoep_size` 仍整除专家数。

`stage3_gather_16bit_weights_on_model_save=false` 时，不进行 16-bit consolidated export；这条路径
是安全 no-op。当前 AutoEP ZeRO-3 不支持直接 consolidated 16-bit export，启用 gather 时会给出
明确错误并引导使用 Universal Checkpoint。

## 9. 代码改动索引

| 文件 | 作用 |
| --- | --- |
| `deepspeed/module_inject/auto_ep_presets/qwen3_vl_moe.py` | 新增 Qwen3-VL 文本 MoE preset |
| `deepspeed/module_inject/auto_ep_presets/registry.py` | 注册新 preset 和 adapter |
| `deepspeed/module_inject/auto_ep_presets/base.py` | 增加 backend/通信配置和嵌套 config 解析接口 |
| `deepspeed/module_inject/auto_ep.py` | 解析模型 config，并生成 AutoEP 参数替换映射计划 |
| `deepspeed/module_inject/auto_ep_config.py` | 解析并校验 `expert_backend`、`serialize_communications` |
| `deepspeed/module_inject/auto_ep_layer.py` | 接入 MUSA TE experts 和串行 All-to-All |
| `deepspeed/moe/ep_experts_musa.py` | MUSA TE GroupedGEMM forward/backward 和 SwiGLU |
| `deepspeed/moe/ep_repack.py` | ZeRO gather、融合权重切片和布局转换 |
| `deepspeed/moe/ep_router.py` | MUSA 2.7.x 稳定 top-k |
| `deepspeed/moe/ep_count.py` | MUSA 2.7.x 可信 host bincount 和完整性检查 |
| `deepspeed/runtime/comm/autoep_serialization.py` | 进程内 collective sequencer |
| `deepspeed/comm/comm.py` | 将 DeepSpeed communication facade 纳入 sequencer |
| `deepspeed/runtime/zero/partitioned_param_coordinator.py` | 按 process group 拆分并排序 ZeRO AllGather |
| `deepspeed/runtime/zero/stage3.py` | 将 ZeRO 梯度归约纳入通信串行边界 |
| `deepspeed/runtime/engine.py` | 在 ZeRO 初始化前重绑定 model parameters/client optimizer，并处理专家保存加载 |
| `deepspeed/checkpoint/autoep_zero3_metadata.py` | 校验真实专家参数名并兼容旧格式 |
| `deepspeed/checkpoint/autoep_universal.py` | Universal Checkpoint 支持后端参数名 |
| `deepspeed/checkpoint/ds_to_universal.py` | 转换时动态生成专家参数 pattern |
| `docs/_pages/config-json.md` | 新配置字段说明 |
| `docs/code-docs/source/autoep.rst` | AutoEP preset 列表 |
| `tests/unit/v1/moe/test_autoep_unit.py` | 配置、路由、MUSA TE、ZeRO-3、通信和 checkpoint 测试 |

## 10. 环境准备和启动检查

### 10.1 已验证软件边界

当前实测环境包括：

- PyTorch 2.7.1；
- torch_musa 2.7.1；
- MUSA 兼容 Transformer Engine；
- Transformers 5.2.0；
- MCCL 2.11.4；
- Qwen3-VL-30B-A3B-Instruct；
- 4 Pod × 8 GPU，world size 32；
- BF16、micro-batch 1、GA=8。

这些版本不是永久约束，但离开该组合后必须重新验证版本门和兼容 workaround。

### 10.2 确认实际导入的是开发目录

多 Pod 环境最常见的问题之一，是修改了 `/home/DeepSpeed`，训练却导入另一个 site-packages。
在每个 Pod 启动前检查：

```bash
python -c 'import deepspeed; print(deepspeed.__file__)'
python -c 'import torch; print(torch.__version__)'
python -c 'import torch_musa; print(torch_musa.__version__)'
```

预期 `deepspeed.__file__` 指向本次同步的开发工作目录。四个 Pod 的关键源文件和训练配置还应
使用 SHA256 校验一致。

### 10.3 检查 TE 后端依赖

```bash
python - <<'PY'
from deepspeed.moe.ep_experts_musa import is_musa_te_grouped_gemm_available
print(is_musa_te_grouped_gemm_available())
PY
```

返回 `True` 只表示依赖可以导入，不等于真实 kernel 已编译、数值正确或性能已经验证。

### 10.4 推荐启动顺序

1. 单卡/EP size 1 做模块和数值冒烟测试；
2. GA=1 做 4 节点短测，检查通信、loss、grad norm 和所有 rank 显存；
3. 至少预热 2 step 后比较稳态性能；
4. 再切换 GA=8 做正式拓扑验证；
5. 长时间运行前验证 checkpoint save/load 和恢复。

## 11. 验证结果和当前状态

### 11.1 单元和路由测试

当前测试覆盖：

- 配置解析、非法 backend 和布尔字段校验；
- Qwen3-VL 只检测文本骨干的 48 个 MoE 层；
- ZeRO Init 下源参数 gather 和专家权重 repack；
- MUSA TE 融合参数布局、零 token 和 dispatch padding；
- stable top-k 和 MUSA 2.7.x 版本门；
- `ep_count` host fallback 版本门；
- 已提前创建的 client optimizer/model-parameter list 在 AutoEP 替换后正确重绑定；
- optimizer 组属性保留，以及已有 state 时原子失败；
- 通信 sequencer 的嵌套、event 和 Work.wait；
- ZeRO-3 process group、梯度归约、checkpoint metadata 和 Universal conversion。

此前完整 AutoEP 单测文件为 75 个通过、1 个失败；失败项是既有的 CPU Mixtral router-logit
capture 对比，与本次 MUSA stable top-k/count 路径无关。本次修复另外增加 client optimizer
重绑定的单元和分布式集成测试，并要求修改文件通过 pre-commit。

### 11.2 client optimizer 修复后的 GA=8 短训对比

EXP83 使用预先创建的 MUSA FusedAdamW、AutoEP size 8、micro-batch 1 和 GA=8。相同数据顺序下，
与最接近的非 AutoEP 历史基线 EXP78 对比如下：

| step | 非 AutoEP loss/grad norm | AutoEP loss/grad norm |
| ---: | ---: | ---: |
| 1 | 1.020 / 13.88 | 1.020 / 13.82 |
| 2 | 1.026 / 13.50 | 1.026 / 13.28 |
| 3 | 1.009 / 13.69 | 1.012 / 13.64 |
| 4 | 1.024 / 13.69 | 1.024 / 13.77 |
| 5 | 1.016 / 13.56 | 1.017 / 13.75 |
| 6 | 1.006 / 13.50 | 1.007 / 13.20 |

六步 loss 最大绝对差为 0.003，平均值相差约 0.08%；grad norm 最大相对差约 2.22%。
这与修复前 AutoEP 约 11.7--12.1 的偏低 grad norm 不同，说明 Router 和专家参数已经重新进入
optimizer/ZeRO-3 更新路径。该结果是短训正确性证据，仍不能替代完整收敛验证。

排除首步编译后，EXP83 step2--6 平均约 237.6 秒，EXP78 对应历史区间约 265.8 秒，短测提升约
10.6%。动态样本长度会影响 step time，因此正式性能结论仍应使用更长时间的配对统计。

### 11.3 optimizer 参数身份验收

使用已创建 client optimizer 的 AutoEP 初始化必须满足：

```text
set(id(p) for current trainable model parameters)
    ==
set(id(p) for optimizer param_groups)
```

同时不能存在重复参数或仍指向原 Hugging Face MoE 层的 stale 参数。分布式集成测试覆盖了
optimizer 对象和组属性保留；单元测试覆盖正常重绑定及源参数已有 state 时不修改任何 param-group
的原子失败行为。

### 11.4 显存状态

EXP83 的一次全 32 卡物理显存采样范围约为 42,530--47,620 MiB / 81,920 MiB，最坏卡仍有约
34.3 GiB 空间。该数据是实时采样而非完整峰值；MoE token 分布和动态序列仍会造成 rank 间差异，
正式长训应继续记录所有 rank 的物理显存峰值和 allocator peak。

## 12. 常见问题排查

### 12.1 配置了 AutoEP，但速度和普通训练相同

依次检查：

1. `expert_parallel.enabled` 是否为 `true`；
2. `preset_model` 是否为 `qwen3_vl_moe`；
3. Transformers 是否至少为 5.2.0；
4. 实际导入的 DeepSpeed 是否来自开发目录；
5. `expert_backend` 是否显式为 `musa_te`；
6. 日志是否显示 48 个文本 MoE 层被替换；
7. 首步是否仍在 TE 编译，是否误用冷启动时间比较。

### 12.2 训练特别快，但 loss 或梯度不同

优先检查：

- stable top-k 是否在 MUSA/torch 2.7.x 生效；
- expert counts 的总和是否等于 `T * top_k`；
- 是否使用了错误的原生 MUSA `bincount`；
- Router 的 score function、归一化、scale 和 top-k 是否与原模型一致；
- client optimizer 是否在 AutoEP 替换前创建，以及其 param-group 是否已重绑定到新参数；
- optimizer 参数身份集合是否与当前模型全部 trainable 参数严格相等；
- 数据顺序、seed、GA、micro-batch 和 checkpointing 是否完全一致；
- 是否只比较了局部 grad norm。

不要把“异常快”直接解释为优化收益。漏 token、少算专家或通信未完成也会让 step 变快。

### 12.3 多机初始化或首个 backward 卡住

检查：

- 四个 Pod 是否只有一个同配置训练任务；
- world size、node rank、master 地址和端口是否一致；
- 四节点 DeepSpeed 文件哈希是否一致；
- `serialize_communications` 是否为 `true`；
- 所有 rank 是否以相同顺序进入 world/EP/EDP collective；
- ZeRO bucket 是否混入不同 `ds_process_group` 的参数；
- MCCL 日志中的最后一个 collective 类型、group 和 message size。

必要时启用分布式 flight recorder 或 MCCL 诊断，并按 rank 对齐最后完成的 collective，不能只看
rank0 日志。

### 12.4 `musa_te` 启动时报依赖或 shape 错误

确认：

- `torch_musa` 和 MUSA TE 可以在训练 Python 环境中导入；
- 输入和专家权重均在 `musa` device；
- dtype 为一致的 FP16/BF16；
- Transformers 权重布局符合 fused gate-up 约定；
- `num_experts` 能被 `autoep_size` 整除；
- 没有把 module-list 专家误配置为 fused backend。

### 12.5 保存或加载时找不到 `w1/w2/w3`

MUSA TE checkpoint 应记录 `expert_parameter_names=["gate_up_proj", "down_proj"]`。如果新
checkpoint 仍只查找 `w1/w2/w3`，通常表示运行时导入了旧 DeepSpeed，或 metadata/转换脚本没有
同步到同一版本。

### 12.6 为什么 `overlap_comm=true` 但 Trace 中通信仍然串行

这是当前安全配置的预期行为。`serialize_communications=true` 会等待每个 collective 完成，
因此不能用 `overlap_comm` 字段本身推断 timeline 中一定存在通信重叠。

## 13. 回退方式

### 13.1 完全回退 AutoEP

```json
{
  "expert_parallel": {
    "enabled": false
  }
}
```

这会保留原 Hugging Face MoE 计算路径，是数值排查时最清晰的基线。

### 13.2 回退 MUSA TE 专家后端

```json
{
  "expert_parallel": {
    "enabled": true,
    "expert_backend": "auto"
  }
}
```

这仍使用 AutoEP，但专家计算回到通用后端。切换后参数布局、显存和性能都可能变化，必须重新做
checkpoint 和数值验证。

### 13.3 关闭通信串行

```json
{
  "expert_parallel": {
    "serialize_communications": false
  }
}
```

当前 MUSA/MCCL 环境不建议直接用于正式训练。它只适合作为明确控制变量的通信并发实验，且必须
配合多 rank Trace、hang watchdog、buffer 生命周期审计和长时间稳定性验证。

stable top-k 和可信 `ep_count` 是 MUSA/torch 2.7.x 的正确性保护，不提供关闭开关。

## 14. 后续工作建议

按正确性和风险优先级排序：

1. 用更长的 GA=8 训练确认 AutoEP 与非 AutoEP 的收敛轨迹；
2. 在长序列和更多 step 下持续验证所有 rank 显存峰值；
3. 为 MUSA 提供经过生产 shape 验证的 device-side expert count，消除 host bincount 同步；
4. 推动 TE/MATE 接受 device `m_splits`/offsets，减少 counts 的 D2H 同步；
5. 在 MCCL 多进程组顺序和 stream 语义明确后，分阶段恢复安全通信重叠；
6. 完成 checkpoint save/load、Universal conversion 和恢复训练的长时间验证；
7. 若要探索 DeepEP，将其作为独立通信后端项目，不与上述正确性修复混合验证。

## 15. 给初学者的验收清单

准备把这套适配用于新环境时，至少逐项确认：

- [ ] 每个 Pod 导入同一份 DeepSpeed 代码；
- [ ] Transformers 版本满足 Qwen3-VL preset 要求；
- [ ] 128 个专家、top-k 8 和 `text_config` 解析正确；
- [ ] `autoep_size` 整除专家数；
- [ ] 每个 EP rank 只持有期望的 16 个专家；
- [ ] MUSA TE forward、dX、dW 都真正执行；
- [ ] client optimizer/model_parameters 已重绑定，且参数身份集合与当前模型一致；
- [ ] stable top-k tie case 结果固定；
- [ ] expert count 总和严格等于 assignment 数；
- [ ] zero-token expert 和动态 shape 数值正确；
- [ ] world、EP、EDP process group 及 collective 顺序一致；
- [ ] GA=1 多机短测无 hang、OOM 和 MCCL timeout；
- [ ] 多步 loss 和各类梯度与非 AutoEP 基线对齐；
- [ ] 记录所有 rank 峰值显存，而不是只看 rank0；
- [ ] 排除首步编译后再比较稳态性能；
- [ ] checkpoint 能保存、加载并继续训练；
- [ ] 明确当前配置的回退方式。

完成这份清单后，才能把“代码可以启动”升级为“适配可以用于训练”；只有进一步通过长时间数值、
性能、显存和稳定性验收后，才能考虑默认启用。
