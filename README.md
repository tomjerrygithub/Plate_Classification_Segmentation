# End-to-end Model To Classify  ‘Single&Double’ car plate and segment the double plate

端到端“单双行车牌”分类及分割模型

## Step 1:样本准备
img_file_name*cls*y1*y2    cls为分类label，例如单行0，双行1，y1,y2 为线条在x=0，x=w处y相对于h的比值
regression：直接回归两个y点；
classification：分别分类两个y点，比如将h划分成32个格子。
数据增强采用有付代码

## Step 2:构造网络和损失函数
搭建网络、最开始采用resnet152，效果非常好，不过训练和推理的速度都较慢；
后改为resnet18、shufflenetv2等轻量级网络，效果仍旧非常好，速度快很多；
构造损失函数，多任务方式，参考yolov3，classification loss + regression loss（两个y 的classification loss）

## Step 3:展开训练
sgd 或 adam 初始lr 0.001 训练20epeoch 效果就比较好了（数据总量在10w张左右，增强后）

## 
