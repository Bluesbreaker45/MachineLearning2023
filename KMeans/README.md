# 实验报告

## K-Means聚类算法原理

K-Means算法是一种无监督分类算法，希望将数据集聚类成 $k$ 个簇$(C=C_{1}, C_{2}, ..., C_{k})$，最小化损失函数为**各个聚类内数据点方差之和**。

要找到以上问题的最优解需要遍历所有可能的簇划分，在实际应用场景中不现实。

由于K-Means算法使用贪心策略求得一个近似解的收敛速度极快，所以实际应用场景中主要采用这种方法。具体步骤如下：

1. 在样本中随机选取 $k$ 个样本点充当各个簇的中心点$μ_1,μ_2,…,μ_k$；
2. 计算所有样本点与各个簇中心之间的距离$dist(x^{(i)},μ_j)$，然后把样本点划入最近的簇中$x_{(i)}∈μ_{nearest}$；
3. 根据簇中已有的样本点，重新计算簇中心 $(\mu_{i}:=\frac{1}{|C_{i}|}\sum_{x\in{C{i}}}x)$；
4. 重复2、3直至达成迭代终止条件。

其中可以根据场景定制的部分有：步骤1中的随机选取初始中心的方法；样本点到簇中心的计算距离方法；求簇中心的方法；迭代终止条件。

## 代码解释

**代码特点**：灵活配置距离计算方法、中心计算方法、迭代停止条件。

完整代码请见文件`./KMeans.ipynb`，各个步骤在代码定义类方法`KMeans.cluster(arr)`中有注释体现。

其中，构造参数的各个含义为

`seedFunc`代表选取初始聚簇中心的方法，默认值为随机选取；

`distanceFunc`代表计算样本点距离的方法，默认为欧氏距离；

`centralPosition`为计算集合内中心的方法；

`maxPass`代表最大迭代次数；

`changeThreshold`代表多少个样本点改变聚簇类才会触发下一轮迭代。

```python
class KMeans:
    def __init__(
            self,
            k=4,
            seedFunc=randSeed,
            distanceFunc=distance,
            centralPosition=meanPos,
            maxPass=100,
            changeThreshold=1
            ):
        self.k = k
        self.distanceFunc = distanceFunc
        self.centralPosition = centralPosition
        self.maxPass = maxPass
        self.changeThreshold = changeThreshold
        self.seedFunc = seedFunc
    
    def cluster(self, arr):
        passNum = 1
        print("pass: ", 1)

        # step 1: randomly choose k seed centroids
        rand = self.seedFunc(range(0, len(arr)), self.k)
        centroids = [makePair(arr[i]) for i in rand]
        clusters = {}
        for c in centroids:
            clusters[c] = set()

        # step 2: assign every points into the set of the closest centroids.
        for i in range(len(arr)):
            c = min(centroids, key=lambda c: self.distanceFunc(arr[i], c))
            clusters[c].add(i)
        
        changed = True
        while changed and passNum < self.maxPass:
            passNum += 1
            print("pass: ", passNum)
            changeNum = 0
            changed = False
            # step 3: calculate new centroids
            newCentroids = [self.centralPosition(clusters[c], arr) for c in centroids]

            # get old centroid-point pairs
            oldPairs = []
            for i in range(len(centroids)):
                oldSet = clusters[centroids[i]]
                oldPairs += [(index, newCentroids[i]) for index in oldSet]

            # step 2: assign every points into the set of the closest centroids.
            # new clusters
            newClusters = {}
            for newC in newCentroids:
                newClusters[newC] = set()
            
            centroids = newCentroids
            clusters = newClusters

            # assign each index to new labels
            for i, oldCentroid in oldPairs:
                c = min(centroids, key=lambda c: self.distanceFunc(arr[i], c))
                clusters[c].add(i)

                if c != oldCentroid:
                    changeNum += 1
            
            if changeNum >= self.changeThreshold:
                changed = True
        
        if not changed:
            print("Iteration stops because change num did not exceed the threshold({}) in the last pass.".format(self.changeThreshold))
        if passNum >= self.maxPass:
            print("Iteration stops because pass num exceeds max pass limit({}).".format(self.maxPass))

        result = []

        # part of step 3: calculate the final centroids
        for centroid in centroids:
            finalCentroid = self.centralPosition(clusters[centroid], arr)
            result += [(index, finalCentroid) for index in clusters[centroid]]

        # get flatten results in the original pixel order
        result.sort(key=lambda p: p[0])

        return centroids, result


kmeans = KMeans(k=4, maxPass=100)
```



## 图像可视化与结果展示

实验输入图像选用课件样例图片的截图（`./输入.png`）：

![](./输入.jpg)

聚类参数选取类构造函数中的默认参数：

```python
kmeans = KMeans(
    k=4, 
    seedFunc=randSeed, 
    distanceFunc=distance, 
    centralPosition=meanPos, 
    maxPass=100, 
    changeThreshold=1
)
```

输出结果如下：

![](./输出(k=4).png)

当聚类数$k=2$时，结果如下（耗时约12s）：

![](./输出(k=2).png)

有时由于聚类数$k$选择较大，造成迭代较慢，可以选择通过设置`maxPass`或`changeThreshold`来通过牺牲很小的精度达到更快的速度。下图为$k=10, maxPass=5$时的聚类结果，耗时约45s，如果不设置maxPass则需要约300s才能跑出结果（实际时由于种子选取的随机性导致了不同运行间不具有严格可比性，在这里仅作示例）：

*设置maxPass=5，实际使用5 passes*

![输出(k=10)](./输出(k=10,maxPass=5,45s).png)

*无设置maxPass数，实际使用39 passes*

![输出(k=10)](./输出(k=10).png)