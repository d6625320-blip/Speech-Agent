# 语音情感专家训练复现

本流程冻结预训练编码器，只训练分类头。请先按 [资源说明](DATA_AND_MODELS.md)自行下载 MCAE-SPPS、中文朗读数据及完整 emotion2vec 权重。

以下命令均从仓库根目录执行，使用 Python 3.10 主环境。主环境安装后补充训练依赖：

```bat
python -m pip install -r Speech_model/requirements.txt
```

所有训练 YAML 中的相对路径以 `Speech_model` 为基准；不需要复制任何历史绝对路径清单。

## 1. MCAE 清单与审计

```bat
python Speech_model/src/prepare_mcae_manifest.py --config Speech_model/configs/data_audit.yaml
python Speech_model/src/audit_mcae_large.py --config Speech_model/configs/data_audit.yaml --no-export-audio
```

第一步生成供审计读取的最小清单，默认拒绝覆盖已有清单；确需重建时可显式加 `--overwrite`。第二步核对元数据、音频和说话人信息，输出主清单及划分。省略 `--no-export-audio` 会额外导出抽样试听文件。

## 2. 特征、LOSO 与基线分类头

```bat
python Speech_model/src/extract_mcae_large_embeddings.py --config Speech_model/configs/feature_extraction.yaml
python Speech_model/src/train_mcae_large_loso.py --config Speech_model/configs/loso_class_balanced.yaml
python Speech_model/src/train_mcae_large_final.py --config Speech_model/configs/final.yaml
```

LOSO 每折按说话人分开训练、验证与测试。最终基线使用全部 MCAE embedding，训练轮数按 `final.yaml` 固定，并读取 LOSO 折外预测拟合温度。不要以外部测试成绩重新选择轮数。最终输出：

```text
Speech_model/outputs/final/mcae_large_class_balanced_ce/final_model.pt
```

可选普通交叉熵对照配置为 `loso_baseline.yaml`。该路径输出位置独立于 class-balanced 实验。

## 3. 中文朗读联合训练

```bat
python Speech_model/src/build_chinese_reading_adaptation_manifest.py --config Speech_model/configs/chinese_reading_joint.yaml
python Speech_model/src/extract_chinese_reading_embeddings.py --config Speech_model/configs/chinese_reading_joint.yaml
python Speech_model/src/train_mcae_chinese_reading_joint.py --config Speech_model/configs/chinese_reading_joint.yaml --validate-only
python Speech_model/src/train_mcae_chinese_reading_joint.py --config Speech_model/configs/chinese_reading_joint.yaml
```

训练使用 MCAE 与中文朗读特征的加权采样，默认比例为 75% / 25%。中文朗读的固定说话人划分为：

| 划分 | 说话人 | 预期样本数 |
|---|---|---:|
| train | SPK0001、0002、0003、0006、0007、0008 | 720 |
| validation | SPK0004、0009 | 240 |
| test | SPK0005、0010 | 240 |

模型仍输出七类，中文朗读集没有中性类。默认温度校准关闭，不把同一验证集上的再拟合结果当成独立证据。

脚本会记录编码器及音频来源指纹、划分、采样情况和训练曲线。已有结果目录默认拒绝覆盖。`--reset-results` 会清理该实验输出，仅在确需重新训练时显式使用，首次复现不需要。

## 4. 评估并接入应用

确定配置后执行：

```bat
python Speech_model/src/evaluate_mcae_chinese_reading_joint.py --config Speech_model/configs/chinese_reading_joint.yaml
```

联合分类头输出为：

```text
Speech_model/outputs/final/mcae_chinese_reading_joint_v1/final_model.pt
```

此路径与应用默认 `speech.checkpoint` 一致。保留完整 checkpoint，不只导出线性层参数。应用还需要预训练 emotion2vec 目录。

外部英语评估可选执行：

```bat
python Speech_model/src/evaluate_mcae_large_final_ravdess.py --config Speech_model/configs/final.yaml
```

该命令评估 MCAE 基线头。测试集只用于最终报告；如果此前看过成绩或用它调整过方案，应披露测试使用历史。

`chinese_reading_cost_sensitive.yaml` 是另一个可选实验配置，需先完成基线和联合训练。它提供特定混淆的代价设置，不代表已证实的情绪距离或性能改善。普通复现不必运行它。

## 5. 复现范围

源码没有附带作者的模型、特征、历史报告或原始数据。随机种子固定也不保证不同 GPU、依赖版本的逐位一致。请记录数据版本、环境、配置和自己的实测结果，不把仓库中模板当作实验成绩。
