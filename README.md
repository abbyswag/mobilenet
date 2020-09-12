# MobileNets
##Differencis
- Первая версия выделилась тем, что впервые стала использовато depth wise separable conv. С помощью этого введения получилось значительно уменьшить размер и сложность модели, меньшее количество параметров и вычеслений сделали Mobilenet особенно полезной для мобильных и встроенных приложений.
- MobileNetV2, основываясь на идеях первой версии, кроме этого вводит в архитектуру linear bottlenecks между слоями и short connections, которые позволяют ускорить обучение и повысить точность
- Последняя, 3 версия, добавила squeeze and excitation слои в изначальные блоки, предсталенные в V2. Согласно авторам статьи, ,лагодаря использованию SE и h-swish в слоях, где тензоры меньше получается меньшая задержка и прирост качества. 

- [ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications (https://arxiv.org/pdf/1704.04861.pdf)](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks  (https://arxiv.org/pdf/1801.04381.pdf)](https://arxiv.org/pdf/1801.04381.pdf)
- [ Searching for MobileNetV3 (https://arxiv.org/pdf/1905.02244.pdf)](https://arxiv.org/pdf/1905.02244.pdf)
Информация также изучалась на таких источниках, как:
-[https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa](https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa)
-[ https://medium.com/@lixinso/mobilenet-c08928f2dba7#:~:text=o%203.2%25%20more%20accurate%20on,MobleNet%20V2%20on%20COCO%20detection.&text=o%206.6%25%20more%20accurate%20compared,MobileNetV2%20model%20with%20comparable%20latency](https://medium.com/@lixinso/mobilenet-c08928f2dba7#:~:text=o%203.2%25%20more%20accurate%20on,MobleNet%20V2%20on%20COCO%20detection.&text=o%206.6%25%20more%20accurate%20compared,MobileNetV2%20model%20with%20comparable%20latency.)
 #Notes
-The requirements.txt file should list all libraries that you need:
```
pip install -r requirements.txt
```
- to run training use:

```
python train_run.py 
       -v v1/v2/v3 , default='v3'
       --mode train/test defolt = train
       --load True - for loading model
       --mobilenet mobilenetv3.pth -pretrained model
       -b, --batch_size, default=64- Batch size for training
       --num_workers, default=8, Number of workers used in dataloading
       --lr, '--learning-rate' default=0.01, - initial learning rate
       -epoch, --max_epoch default=100, - max epoch for training
       --save_folder, default='img/'
       --save_img, default=True, save test images
       --weight_decay, default=5e-4, Weight decay for SGD
       --momentum', default=0.999
```
For example:
```
python train_run.py -v v1 --mode test --load True --mobilenet mobilenetv1.pth
```
- to run validation and save image in folder:
```
python train_run.py -v v2 --mode test --load True --mobilenet mobilenetv2.pth
```
