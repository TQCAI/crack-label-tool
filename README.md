- 项目地址：  https://github.com/TQCAI/crack-label-tool
- 博客介绍：  https://blog.csdn.net/TQCAI666/article/details/89378837

# 界面区域

![界面区域](https://img-blog.csdnimg.cn/20190418144205691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RRQ0FJNjY2,size_16,color_FFFFFF,t_70)
# 标注模式



- `edge detect`
主要采用边缘检测算法对裂缝进行预标注：首先用中值滤波对图片进行平滑处理，之后用canny算子进行边缘检测，并用形态学方法进行了膨胀、小连通块去除，得到了大致的识别图像供标注者处理。
- `empty`
不进行预识别，让标注者完成所有的标注工作。
- `keep annote`
读取后缀为"**_gt**"的标注文件，用于对已标注图片的修改。
- `FCN`
启用此模式需要在代码中进行配置。启用后通过部分训练的FCN算法模型对裂缝进行预标注。

## 边缘检测
- 原图


<img src="https://img-blog.csdnimg.cn/20190418145253875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RRQ0FJNjY2,size_16,color_FFFFFF" width="50%"  />



- 预标注图


<img src="https://img-blog.csdnimg.cn/20190418145320441.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RRQ0FJNjY2,size_16,color_FFFFFF" width="50%"  />
