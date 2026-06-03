整体结构是合理的，思路清晰，三个模块的开关设计也很干净。但有几个值得关注的问题：

## 主要问题

### 1. RoPE 的接入方式有问题

你把 `rope_cos/sin` 作为 `extra_kwargs` 传给 `trm_encoder`，但 RecBole 原生的 `TransformerEncoder` 和 `MultiHeadAttention` 并不接受这些参数。这意味着：

- 如果没有修改 RecBole 内部层，`**extra_kwargs` 会被忽略，RoPE 根本没生效
- RoPE 应该在 `MultiHeadAttention` 内部、`Q/K` 做 matmul 之前施加，而不是在外部传入

正确做法是要么**自定义 TransformerEncoder**，要么继承并 override attention 层。同理 `time_bias` 也需要在 attention score 计算处加入（`QK^T / sqrt(d)` 之后，softmax 之前）。

### 2. `RotaryPositionEmbedding` 和 `compute_time_bias` 在 RecBole 里不存在

```python
from recbole.model.layers import (
    RotaryPositionEmbedding,
    compute_time_bias,
)
```

RecBole 标准库没有这两个组件，这会直接 `ImportError`。需要自己实现并放到合适位置。

### 3. 对比损失的梯度问题（hierarchical 模式）

```python
pos_mask = torch.zeros(...)
pos_mask = pos_mask.masked_fill(same_l3 & not_self, 1.0)
# ...
```

`pos_mask` 是纯 tensor 权重，不参与梯度，这没问题。但 `has_pos` 做索引后求 mean，如果某个 batch 里 item 类别非常稀疏，`has_pos` 为空时会 nan——你的 `if pos_mask.sum() == 0` 只检查了 binary 的情况，hierarchical 模式下也需要检查。

### 4. `calculate_loss` 返回值不一致

```python
if self.use_contra_loss:
    return main_loss, contra_loss   # tuple
return main_loss                    # scalar
```

RecBole 的 trainer 期望 `calculate_loss` 返回单个 scalar。返回 tuple 会导致 trainer 里的 `loss.backward()` 报错。应该直接返回 `main_loss + contra_loss`。

### 5. 对比损失计算范围太大

```python
items = torch.unique(item_seq[item_seq > 0])
embs = self.item_embedding(items)  # [N, H]
sim = embs @ embs.T  # [N, N]
```

一个 batch 里 unique item 可能有几千个，`[N, N]` 的 sim 矩阵会很大，而且大量 item pair 是负样本，梯度信号稀疏。更常见的做法是只对 `seq_output`（序列表示）或 pos item embedding 做 InfoNCE，而不是对所有出现过的 item。

---

## 次要建议

| 问题 | 建议 |
|---|---|
| `precompute` 在 CPU 上做，然后 `register_buffer` | 没问题，但注意 `device="cpu"` 后 forward 里需要 `.to(item_seq.device)`，或者直接不指定 device |
| `self.time_bias_weight = nn.Parameter(torch.tensor(0.01))` | 初始值 0.01 很小，可以考虑加个 `nn.Tanh` 或 clamp 防止它变成很大的负数破坏 attention |
| `SASRecF` 里的 `position_embedding` 用 `del` 删除 | 更安全的方式是 `self.position_embedding = None` 并在 forward 判断，`del` 会让 `state_dict` 加载出问题 |

---

## 总结

核心逻辑上 RoPE 和 time bias **没有真正接入 attention 计算**，这是最需要解决的问题。对比损失的返回值类型和 hierarchical 下的空检查也会直接导致训练崩溃。建议先实现一个自定义的 `MultiHeadAttentionWithRoPE`，把 `rope_cos/sin` 和 `time_bias` 都在那里消费掉，再接入 encoder。