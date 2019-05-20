
# 100 numpy exercises

numpy メーリングリスト、 stack overflow、numpy ドキュメントから集めた numpy 演習です。この演習の目標は、新旧両方のユーザ向けクイックリファレンスとして提供することだけではなく、これらの人に指導する人向けの演習を提供することです


もしエラーに気づいたりあなたがより良い解決方法を思いついたら、自由にIssueを開いてください。<https://github.com/rougier/numpy-100>

#### 1. `np` という別名で numpy モジュールをインポートしてください。 (★☆☆)


```python
import numpy as np
```

#### 2. numpy のバージョンと設定を表示する。 (★☆☆)


```python
print(np.__version__)
np.show_config()
```

#### 3. 各要素を0とするサイズが10のベクトルを作成する。 (★☆☆)


```python
Z = np.zeros(10)
print(Z)
```

#### 4. 配列のメモリサイズを求める。 (★☆☆)


```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```

#### 5. コマンドラインから numpy の add 関数のドキュメントを表示する。 (★☆☆)


```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```

#### 6. 各要素を0とするサイズが10のベクトルを作成する。ただし、5番目の要素を1にして表示してください。 (★☆☆)


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

#### 7. 要素が10から49のベクトルを作成する。 (★☆☆)


```python
Z = np.arange(10,50)
print(Z)
```

#### 8. ベクトルの要素を反転する。(最初の要素を最後の要素にする) (★☆☆)


```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```

#### 9. 0から8までの範囲を値としてもつ 3x3 の行列を作成する。 (★☆☆)


```python
Z = np.arange(9).reshape(3,3)
print(Z)
```

#### 10. \[1,2,0,0,4,0\] から0ではない要素のインデックスを探す。 (★☆☆)


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```

#### 11. 3x3 の単位行列を作成する。 (★☆☆)


```python
Z = np.eye(3)
print(Z)
```

#### 12. 乱数で 3x3x3 の配列を作成する。 (★☆☆)


```python
Z = np.random.random((3,3,3))
print(Z)
```

#### 13. 乱数で 10x10 の配列を作成しその最小値、最大値を計算する。 (★☆☆)


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

#### 14. 要素を乱数、サイズを30とするベクトルを作成し、その平均値を計算する。 (★☆☆)


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

#### 15. 周囲が1で内部が0である2次元配列を作成する。 (★☆☆)


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

#### 16. 既存の配列の周囲を0で埋める。 (★☆☆)


```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
```

#### 17. 以下の式の結果を答える。 (★☆☆)


```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```

#### 18. 対角成分の直下の要素が 1,2,3,4 である 5x5 の行列を作成する。 (★☆☆)


```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```

#### 19. 碁盤の目状で埋めた 8x8 の行列を作成する。 (★☆☆)


```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```

#### 20. (6,7,8) 形状の配列がある場合、1次元配列にしたときの100番目の要素に該当するインデックス(x,y,z)を答える。


```python
print(np.unravel_index(99,(6,7,8)))
```

#### 21. tile 関数を使って碁盤目状の 8x8 行列を作成する。 (★☆☆)


```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```

#### 22. 5x5 のランダム行列を正規化する。 (★☆☆)


```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```

#### 23. 色を4つの符号なしバイト型(RGBA)として表現するカスタムdtypeを作成する。 (★☆☆)


```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```

#### 24. 5x3 の行列に 3x2 の行列を乗算する。(実行列の積) (★☆☆)


```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```

#### 25. 1次元配列を与えられる場合、3よりも大きく8以下である要素を負の数にする。 (★☆☆)


```python
# Author: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)
```

#### 26. 以下のスクリプトの出力を答える。 (★☆☆)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. Zを整数型ベクトルとして、どの式が適切かを答える。 (★☆☆)


```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. 以下の式の結果を答える。


```python
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
```

#### 29. 浮動小数点型の配列を0から離れる方向に丸めて整数にする。 (★☆☆)


```python
# Author: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))
```

#### 30. 2つの配列間で共通の値を探す。 (★☆☆)


```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```

#### 31. numpy の全警告を無視する方法を答える。(非推奨) (★☆☆)


```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)
```

An equivalent way, with a context manager:

```python
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

#### 32. 以下の式は True かを答える。 (★☆☆)


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. 昨日、今日、明日の日付を取得する。 (★☆☆)


```python
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```

#### 34. 2016年7月に対応するすべての日付を取得する。 (★★☆)


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```

#### 35. ((A+B)\*(-A/2)) を計算する。(左記の式をそのまま使用しない。) (★★☆)


```python
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```

#### 36. 5つの方法で乱数(実数)の配列から整数部分を抽出し表示してください。 (★★☆)


```python
Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
```

#### 37. 各行の値が0から4の範囲にある 5x5 の行列を作成する。 (★★☆)


```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
```

#### 38. 10の整数を生成するジェネレータ関数がある場合、それを使って配列を作成する。 (★☆☆)


```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```

#### 39. 0から1の範囲(両端の0と1は含めない)にあるサイズが10のベクトルを作成する。 (★★☆)


```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```

#### 40. サイズが10である乱数のベクトルを作成し、それをソートする。 (★★☆)


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```

#### 41. np.sum よりも高速に小さい配列を集計する。 (★★☆)


```python
# Author: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```

#### 42. 2つの乱数の要素でできた配列 A、Bがある場合、それらが等しいかを判定する。 (★★☆)


```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)
```

#### 43. イミュータブルな配列(読み取り専用)を作成する。 (★★☆)


```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```

#### 44. 直行座標を表現する 10x2 の乱数行列がある場合、それらを極座標に変換する。 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```

#### 45. サイズが10である乱数のベクトルを作成し、最大値を0に置換する。 (★★☆)


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```

#### 46. \[0,1\]x\[0,1\] 領域を網羅する `x` と `y` 座標をもつ構造化配列を作成する。 (★★☆)


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```

####  47. 2つの配列 X 、 Y を与えられる場合、コーシー行列 C (Cij =1/(xi - yj)) を作成する。


```python
# Author: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```

#### 48. numpy の各スカラー型について表示可能な最小値、最大値を表示する。 (★★☆)


```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```

#### 49. 配列のすべての値を表示してください。 (★★☆)


```python
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)
```

#### 50. ベクトルにおいて(与えられたスカラーに)最も近い値を探す。 (★★☆)


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

#### 51. 位置 (x,y) 、色 (r,g,b) を表現する構造化配列を作成する。 (★★☆)


```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```

#### 52. 座標を表す (100,2) 形状の乱数配列がある場合、全ての点の距離を計算する。 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```

#### 53. 浮動小数点 (32 bits) 型配列を整数 (32 bits) 型配列に変換する。


```python
Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
print(Z)
```

#### 54. 以下のファイル読み込む。 (★★☆)


```python
from io import StringIO

# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```

#### 55. numpy 配列を enumerate する。 (★★☆)


```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```

#### 56. 一般的なガウス(正規)分布の2次元配列を作成する。 (★★☆)


```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```

#### 57. 2次元配列で p つの要素をランダムに配置する。 (★★☆)


```python
# Author: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```

#### 58. 行列の各行の平均値を減算する。 (★★☆)


```python
# Author: Warren Weckesser

X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```

#### 59. 配列をn番目の列をキーにソートする。 (★★☆)


```python
# Author: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```

#### 60. 特定の2次元配列においてすべて0である列が存在するかを判定する。 (★★☆)


```python
# Author: Warren Weckesser

Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
```

#### 61. 配列において与えられる値に最も近い値を探す。 (★★☆)


```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```

#### 62. (1,3) 、 (3,1) の形状をもつ2つの配列がある場合、それらをイテレータを使って集計する。 (★★☆)


```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```

#### 63. name属性を持つ配列クラスを作成し表示してください。 (★★☆)


```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```

#### 64. 特定のベクトルがある場合、もう一つのベクトルで指定されたインデックスに対応する各要素に1を加算する。(インデックスは繰り返し指定されることに気を付ける。) (★★★)


```python
# Author: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```

#### 65. インデックスリスト (I) を基にベクトル (X) の要素を配列 (F) に累積する。 (★★★)


```python
# Author: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```

#### 66. (w,h,3) (dtype=ubyte) の画像がある場合、ユニークな色の数を計算する。 (★★★)


```python
# Author: Nadav Horesh

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))
```

#### 67. 4次元配列がある場合、一度に最後の2軸を合計する。 (★★★)


```python
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```

#### 68. 1次元のベクトル D がある場合、その部分集合のインデックスを表現する同じサイズのベクトル S を使って、D の部分集合の平均を計算する。 (★★★)


```python
# Author: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```

#### 69. ドット積の対角要素を取得する。 (★★★)


```python
# Author: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version  
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)
```

#### 70. ベクトル \[1, 2, 3, 4, 5\] がある場合、各要素間に3つの0を挟んだ新しいベクトルを作成する。 (★★★)


```python
# Author: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

#### 71. (5,5,3) の配列がある場合、(5,5)の配列ごとに乗算する。。 (★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```

#### 72. 配列のうち2行の値を交換する。 (★★★)


```python
# Author: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```

#### 73. 三角形を表現する10の三つの点の集合(頂点を共有する)がある場合、すべての三角形を構成するユニークな線分の集合を探す。 (★★★)


```python
# Author: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

#### 74. bincountで作成された配列 C が与えられる場合、np.bincount(A) == C となるような配列 A を作成する。 (★★★)


```python
# Author: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```

#### 75. スライディングウィンドウを使って配列の平均を計算する。 (★★★)


```python
# Author: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```

#### 76. 1次元配列 Z があるとして、先頭行は (Z\[0\],Z\[1\],Z\[2\])、後続する行は各々 +1 される(最終行は (Z\[-3\],Z\[-2\],Z\[-1\] であるべき)2次元配列を作成する。 (★★★)


```python
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

#### 77. ブール型を論理否定する、または、浮動小数点型の符号を変更する。 (★★★)


```python
# Author: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```

#### 78. 線(2次元)を表現する2組の点 P0、P1 と 点 p があるとして、点 p から 各線 i (P0\[i\],P1\[i\]) までの距離を計算する。 (★★★)


```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```

#### 79. 線(2次元)を表現する2組の点 P0、P1 と 1組の点 P がある場合、各点 j (P\[j\]) から 各線 i (P0\[i\],P1\[i\]) までの距離を計算する。 (★★★)


```python
# Author: Italmassov Kuanysh

# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

#### 80. 任意の配列がある場合、与えられた要素を中心とする固定の形状の部分配列を抽出する。(必要なときは `fill` で値を埋める) (★★★)


```python
# Author: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

#### 81. 配列 Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\] がある場合、配列 R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\] を作成する。 (★★★)


```python
# Author: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```

#### 82. 行列のランクを求める。 (★★★)


```python
# Author: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```

#### 83. 配列の最頻値を探す。


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```

#### 84. 10x10 のランダム行列から隣接する 3x3 の小行列をすべて抽出する。 (★★★)


```python
# Author: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```

#### 85. Z\[i,j\] == Z\[j,i\] となるような2次元配列のサブクラスを作成する。 (★★★)


```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

#### 86. p つの (n,n) 形状の行列と p つの (n,1) 形状のベクトル行列がある。p つの行列積を一度に求める。(結果は、(n,1) 形状となる。) (★★★)


```python
# Author: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

#### 87. 16x16 の配列がある場合、4x4 の小行列ごとの和を求める。 (★★★)


```python
# Author: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```

#### 88. numpy 配列を使ってライフゲームを実装する。 (★★★)


```python
# Author: Nicolas Rougier

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```

#### 89. 配列のうち n 番目に大きな値を取得する。 (★★★)


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])
```

#### 90. 任意の数値をもつ複数のベクトルを与えられる場合、デカルト(直)積(すべての数値のすべての組み合わせ)を求める。 (★★★)


```python
# Author: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

#### 91. 普通の配列から構造化配列を作成する。 (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```

#### 92. 大きなベクトル Z がある場合、3種の方法を使って3乗する (★★★)


```python
# Author: Ryan G.

x = np.random.rand(5e7)

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```

#### 93. (8,3) 形状の配列 A と (2,2) 形状の配列 B がある。B の2行の要素を各々1つ以上ずつ含む A の行を探す。 (★★★)


```python
# Author: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

#### 94. 10x3 の行列がある場合、すべての値が同等ではない行(例えば、 \[2,2,3\]) を抽出する。 (★★★)


```python
# Author: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```

#### 95. 整数型のベクトルを各整数に対応する2進数表現の行列に変換する。 (★★★)


```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

#### 96. 2次元配列を与えられる場合、ユニークな並びの行を抽出する。 (★★★)


```python
# Author: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# Author: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```

#### 97. 2つのベクトル A、B がある場合、inner、outer、sum、and multiply 関数に相当する機能を einsum 関数で記述する。 (★★★)


```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```

#### 98. 2つのベクトル (X,Y) でパスを表現する場合、等間隔のサンプルを使ってそれを補完する。 (★★★)?


```python
# Author: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```

#### 99. 整数 n 、2次元配列 X を与えられる場合、X から母数 n の多項分布を基に描かれたと解釈されうる行(すなわち、整数のみを含み合計が n となる行)を選択する。 (★★★)


```python
# Author: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

#### 100. 1次元配列 X の平均における95%信頼区間をブートストラップ法で求める。(すなわち、配列の要素を N 回リサンプリングし、各サンプルの平均を計算し、それから百分位数を求める。) (★★★)


```python
# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```
