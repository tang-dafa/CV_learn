#  Task1 赛题理解

名称：街道字符识别
赛题任务：以计算机视觉中字符识别为背景，要求选手预测街道字符编码

## 1.数据解读

![image-20200519185413205](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200519185413205.png)

给出字符的位置框

![image-20200519185520305](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200519185520305.png)

数据

![image-20200520100625495](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520100625495.png)

## 2.评测指标

提交结果与实际图片的编码进行对比，以编码整体识别准确率为评价指标。

计算公式：

score=编码识别正确的数量/测试集图片数量

## 3.读取数据

```python
# 数据标注
def parse_json(d):
    # 创建二维数组
    arr = np.array([
        d['top'],d['height'],d['left'],d['width'],d['label']
    ])
    # 数据类型统一为整数型
    arr = arr.astype(int)
    return arr

# 读取图像,多维数组形式保存，前两维表示图片的像素坐标，最后一维表示图片的通道索引
img = cv2.imread(r'D:\\PartGit\\team-learning\\cv_data\\train\\train\\000000.png')
arr = parse_json(train_json['000000.png'])
# print(arr)

plt.figure(figsize=(10,10))
# plt.subplot(2,2,1)表示将整个图像窗口分为2行2列, 当前位置为1
# image.shape[0], 图片垂直尺寸,image.shape[1], 图片水平尺寸,image.shape[2], 图片通道数
# print(arr.shape[1]+1)
plt.subplot(1, arr.shape[1]+1, 1)

plt.imshow(img)

# x,y坐标显示为空
plt.xticks([]);plt.yticks([])

for idx in range(arr.shape[1]):
    print(idx)
    print('******')
    print(arr)
    plt.subplot(1 , arr.shape[1]+1 , idx+2)
    # arr[0,idx]: arr[0,idx]+arr[1,idx] 这个是指顶点加高
    # arr[2,idx]: arr[2,idx]+arr[3,idx] 这个是指顶点加宽
    # 图片的大小
    plt.imshow(img[arr[0,idx]: arr[0,idx]+arr[1,idx],
                    arr[2,idx]: arr[2,idx]+arr[3,idx]])
    plt.title(arr[4,idx])
    plt.xticks([]);plt.yticks([])
```

结果

![image-20200520111112933](C:\Users\dafa\AppData\Roaming\Typora\typora-user-images\image-20200520111112933.png)

## 4.解题思路

查阅数据文件可知图片字符个数不一致，所以赛题难点在于对不定长的字符进行识别。思路有3：

1. ### 定长字符识别

   将所有图像都填充为最长字符来识别

2. ### 专业字符识别思路：不定长字符识别

   CRNN字符识别模型

3. ### 专业分类思路：检测再识别

   首先将字符的位置进行识别，利用物体检测的思路完成。需要参赛选手构建字符检测模型，对测试集中的字符进行识别。可以参考物体检测模型SSD或者YOLO来完成。

### 