## 环境配置
创建conda环境
```shell
conda create -n medClassify python=3.8
```
安装依赖
```shell
pip install -r requirements.txt
```
## 功能
* 在config文件中，可以调节各个参数，具体详见配置文件。
* 提供了部分基础2D和3D分类的算法。
* 目前提供了二维的图片分类和三维CAMUS数据集的分类，如果想自行扩展，请自定义数据集类进行修改。
* 如果您想进行**多分类**任务，请自行修改**num_classes**参数

## 新增模型
添加新的模型，请在get_model方法进行添加。

## 训练
```shell
python main.py
```


## Done
### Network
* 2D
- [x] alexnet
- [x] densenet
- [x] googlenet
- [x] mobilenet
- [x] nasnet
- [x] resnet
- [x] resnext
- [x] vggnet
* 3D
- [x] densenet3d
- [x] resnet3d
- [x] resnext3d
