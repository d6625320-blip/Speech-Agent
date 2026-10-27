# 基线与联合分类头诊断

此脚本读取已有特征和分类头，不训练模型。默认只诊断训练、验证部分：

```bat
python Speech_model/src/evaluate_chinese_reading_baseline.py --config Speech_model/configs/chinese_reading_joint.yaml --split diagnostics
```

只准备 MCAE 基线时加 `--original-only`。正式确定实验配置后才使用 `--split test`。下载、缓存生成与训练见 [训练复现](../docs/TRAINING.md)。本仓库不附带缓存、权重或历史性能数字。
