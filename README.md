# pairwise-audio-occlusion-path

## 领域

游戏、图形与多媒体

## 主题

3D 音频遮挡

## 说明

多点路径、距离分级与混响联动。

初始状态下 `media/audio_occlusion.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
