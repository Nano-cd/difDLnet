![caption_42](https://github.com/user-attachments/assets/2e4ca79a-0598-4859-8e03-1c6a2b9c3bb4)# DIF-DLNet 图像描述生成项目

## 主要改进

### 架构改进
1. 采用新型Decoder架构设计
   - 多层Transformer结构
   - 改进的注意力机制模块
2. 强化学习预训练优化
   - 新增对比学习损失函数
   - 在强化学习阶段前加入预训练

### 可视化升级
- 集成AttVis可视化模块
- 支持注意力热力图生成
- 改进Decoder可视化接口

## 快速开始
```python
from models.diffnet_pro import DIFNetPro
model = DIFNetPro(pretrained=True)
```

```python
热力图生成
python attvis
```

## 模型下载
百度网盘链接：
- 模型权重、Loss曲线：[DIFNetPro_best.pth]([https://pan.baidu.com/s/1abc123def456ghi789jklmn](https://pan.baidu.com/s/1lhIunoMCOC7pYwPARqqXtg?pwd=ktwq)) 提取码：ktwq

## 可视化示例:本文方法
![Attention Heatmap](docs/![caption_42](https://github.com/user-attachments/assets/98c26fd0-9d37-437a-af37-aaec1e293dbd)
)

## 可视化示例：原文权重
![Attention Heatmap](docs/![caption_42](![caption_42](https://github.com/user-attachments/assets/4c69d0b2-873d-4b99-8756-0604d4b8ed6a)
)


## 技术细节
详细改进说明见[TECHNICAL.md](docs/TECHNICAL.md)
