# 语音训练流程

完整命令统一维护在 [训练复现](../docs/TRAINING.md)。配置在当前目录的 `configs/`，不包含其他层级。

流程：自行下载数据及编码器 → 生成 MCAE 清单 → 审计及划分 → 冻结编码器提取特征 → LOSO 和最终基线头训练 → 可选联合训练 → 独立评估 → 部署推理。

模型选择使用验证集；独立测试集不得用于选择门槛、训练轮次或采样比例。英语表演语音测试与真实中文对话应分开报告。

在仓库根运行基线推理：

```bat
python Speech_model/src/predict_mcae_large_final.py --config Speech_model/configs/final.yaml --audio "sample.wav"
```

权重需自行训练。该命令使用 MCAE 基线；联合分类头由应用默认配置加载。必须保留 checkpoint 的标签顺序、特征均值与标准差。
