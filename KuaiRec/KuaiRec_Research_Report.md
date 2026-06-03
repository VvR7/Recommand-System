# KuaiRec 数据集调研报告

> **数据集全称**: KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems
> **来源**: CIKM 2022, 快手 + NUS/四川大学/人大/中科大
> **论文**: [Gao et al., CIKM 2022](https://dl.acm.org/doi/abs/10.1145/3511808.3557220)
> **主页**: https://chongminggao.github.io/KuaiRec/
> **许可证**: CC BY-SA 4.0

---

## 一、数据集概览

KuaiRec 来自快手短视频平台真实推荐日志，是**首个真实世界"完全可观测"（fully-observed）推荐数据集**。其核心特色是提供了一个稠密度高达 **99.6%** 的用户-物品交互矩阵（小矩阵），即几乎所有用户都看过几乎所有视频并留下了反馈，这在传统推荐数据集中极为罕见（通常密度 < 1%）。

### 关键统计

| 指标 | Big Matrix | Small Matrix |
|------|-----------|--------------|
| 用户数 | 7,176 | 1,411 |
| 物品数 | 10,728 | 3,327 |
| 交互数 | 12,530,806 | 4,676,570 |
| 稠密度 | 16.3% | **99.6%** |
| 时间跨度 | 2020-07-05 ~ 2020-09-05 (63天) | 同左 |
| watch_ratio 均值 | 0.945 | 0.907 |
| 每用户平均交互数 | 1,746 | 3,314 |

---

## 二、文件结构与内容详解

### 2.1 核心交互矩阵

#### `big_matrix.csv` (12.5M 行)
大矩阵——真实推荐场景的稀疏交互记录。

| 字段 | 含义 |
|------|------|
| `user_id` | 用户ID (0~7175) |
| `video_id` | 视频ID (0~10727) |
| `play_duration` | 用户实际播放时长 (ms) |
| `video_duration` | 视频总时长 (ms) |
| `time` | 播放时间戳 (精确到毫秒) |
| `date` | 播放日期 (YYYYMMDD) |
| `timestamp` | Unix 时间戳 |
| `watch_ratio` | 观看比例 = play_duration / video_duration |

#### `small_matrix.csv` (4.7M 行)
小矩阵——从大矩阵中筛选出的 1,411 名用户对 3,327 个视频的**近乎全观测**交互。大部分交互的 `time/date/timestamp` 字段为 NaN（表示未自然曝光但用户仍观看了视频，通过实验设计获取）。

### 2.2 用户特征

#### `user_features.csv` (7,176 用户)

| 字段 | 含义 |
|------|------|
| `user_active_degree` | 活跃度: full_active (85%) / high_active (13%) / middle_active |
| `is_live_streamer` | 是否直播作者 |
| `is_video_author` | 是否视频作者 |
| `follow_user_num` | 关注数 |
| `fans_user_num` | 粉丝数 |
| `friend_user_num` | 好友数 |
| `register_days` | 注册天数 |
| `onehot_feat0~17` | 18 个匿名化 one-hot 特征 |

#### `user_features_raw.csv` (7,176 用户)
包含未匿名化的人口统计学特征：性别、年龄段、手机品牌/型号/价格、地理位置（国家/省份/城市/城市等级/社区类型）、安装的竞品APP（抖音/火山/西瓜/斗鱼/虎牙/YY）、设备平台/系统版本、运营商等。

### 2.3 物品特征

#### `item_categories.csv` (10,728 物品)
| 字段 | 含义 |
|------|------|
| `video_id` | 视频ID |
| `feat` | 多标签类别ID列表 (如 `[8]`, `[27, 9]`)，共31个不同类别标签 |

#### `item_daily_features.csv` (343K 行 = 10,728 × 63天)
物品逐日特征，**58个字段**，包括：

- **基础信息**: author_id, video_type, upload_dt, upload_type, visible_status, video_duration, video_width/height
- **曝光与播放**: show_cnt, play_cnt, complete_play_cnt, valid_play_cnt, long/short_time_play_cnt
- **互动指标**: like_cnt, comment_cnt, follow_cnt, share_cnt, download_cnt, collect_cnt, report_cnt 等（及对应去重用户数）
- **音乐/标签**: music_id, video_tag_id, video_tag_name

#### `kuairec_caption_category.csv` (10,728 物品)
| 字段 | 含义 | 覆盖率 |
|------|------|--------|
| `manual_cover_text` | 手动标注封面文字 | 37.6% |
| `caption` | 视频标题/描述文本 | 87.4% |
| `topic_tag` | 话题标签 (JSON数组) | 57.1% |
| `first/second/third_level_category` | 三级类别体系 | 100% |

#### `video_raw_categories_multi.csv` (26,827 行)
多标签视频分类原始数据，包含分类概率 (`prob`)、层级关系 (`root_id→parent_id→category_id`) 及分类来源。

### 2.4 社交网络

#### `social_network.csv` (472 用户)
| 字段 | 含义 |
|------|------|
| `user_id` | 用户ID |
| `friend_list` | 好友ID列表 (如 `[4202, 7126]`) |

**特点**: 覆盖用户仅有 472 人，社交边较稀疏，平均好友数约 2~3 人。

---

## 三、数据集核心特点总结

1. **全观测矩阵 (99.6% 密度)**：小矩阵几乎无缺失值，可作为推荐效果评估的**真实 Ground Truth**，解决曝光偏差问题。
2. **双矩阵设计**：大矩阵模拟真实推荐场景（16.3% 密度，有偏），小矩阵提供全量偏好参考，便于对比研究。
3. **丰富的多模态特征**：文本（中文标题/标签/封面文字）、类别体系（3级层次）、用户画像、物品逐日统计等。
4. **时间维度**：63天连续数据，带精确时间戳，支持序列建模、流行度演化和时序动态分析。
5. **社交关系**：部分用户之间的好友关系图。
6. **真实互动反馈**：watch_ratio 作为连续型偏好信号，比显式评分更自然、更细粒度。


## 四、数据使用注意事项

1. **Small Matrix 的 NaN 时间戳**：表示这些交互并非自然曝光产生，使用时需注意区分有机交互和实验采集交互。
2. **watch_ratio > 1**：部分用户会快进/重复播放，需根据任务决定是否截断或保留。
3. **社交网络稀疏**：仅 472/7176 用户有社交信息，社交推荐方法需考虑覆盖范围。
4. **中文文本**：caption 和 tag 均为中文，使用多语言模型时需注意。
5. **类别噪声**：`-124` 表示 UNKNOWN 类别，多级分类中存在缺失。
6. **时间范围有限**：仅 63 天，长期兴趣周期、季节性效应可能无法充分捕捉。

