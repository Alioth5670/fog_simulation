# 基于深度的图像加雾 (image fog simulation based on depth map)

requirement:
* opencv-python
* numpy
  
run demo
```cmd
python demo.py
```

## Demo

* input:
![alt](./image/input.png 'input')
* output:
![alt](./image/result.png 'result')

## 获取深度图 (get depth image)
* [Depth anything](https://github.com/LiheYoung/Depth-Anything)

## Reference

* Semantic Foggy Scene Understanding with Synthetic Data: [paper](https://github.com/sakaridis/fog_simulation-SFSU_synthetic),[code](https://arxiv.org/abs/1708.07819)
* Single image haze removal using dark channel prior: [paper](https://ieeexplore.ieee.org/document/5206515), [code](https://github.com/Kiumb1223/DefogAlgorithm_DarkChannelPrior)
* guided filter: [paper](https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf), [code](https://github.com/swehrwein/python-guided-filter)
