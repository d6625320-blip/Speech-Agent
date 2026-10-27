# 语音情感专家

冻结 `emotion2vec_plus_large` 编码器，提取 1024 维 utterance embedding，训练独立七分类头。标签顺序：

```text
angry, disgust, fear, happy, neutral, sad, surprised
```

数据集、预训练编码器、特征缓存和分类头均不包含在仓库中。来源及布局见 [数据与模型](../docs/DATA_AND_MODELS.md)，完整命令见 [训练复现](../docs/TRAINING.md)。

保留 MCAE-SPPS 的 LOSO / class-balanced CE 基线，以及 MCAE-SPPS 与中文朗读的联合训练路径。应用默认使用联合分类头，不默认应用温度校准。中文朗读只有六类非中性情绪，不能以其六类成绩推断中性类性能。

`src/` 为数据准备、训练和评估脚本，`configs/` 为相对于本目录的配置，`tests/` 使用合成特征验证实现。`data/`、`models/`、`features/` 和 `outputs/` 在用户下载或运行后建立，均由根 `.gitignore` 排除。
