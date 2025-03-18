# DIF-DLNet 图像描述生成项目

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
from models.diffnet_pro import DIFNetPro
model = DIFNetPro(pretrained=True)
```

## 模型下载
百度网盘链接：
- 模型权重：[DIFNetPro_best.pth](https://pan.baidu.com/s/1abc123def456ghi789jklmn) 提取码：8888
- Loss曲线：[训练loss记录曲线](https://pan.baidu.com/s/1xyz987uvw654rst321ponm) 提取码：6666

## 可视化示例
![Attention Heatmap](docs/heatmap_example.png)

## 技术细节
详细改进说明见[TECHNICAL.md](docs/TECHNICAL.md)
