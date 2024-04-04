### 语言元素
#### 指令和程序

#### 变量和类型
变量是存储数据的载体，（变量和数据的关系），而不同的数据需要定义不同的类型去存储。<br>
数据类型：
- int
- float: 科学计数法
- string: 以`单`引号或双引号括起来的任意文本，比如`'hello'`和`"hello"`
- bool
	
#### 变量命名
1. Start: NO !NUMBERS！--> letter numbers underline
2. lower letters connect with '_'

#### 类型检查
	- type函数

``` Python
"""
使用type()检查变量的类型

"""
a = 100
b = 12.345
c = 1 + 5j
d = 'hello, world'
e = True
print(type(a))    # <class 'int'>
print(type(b))    # <class 'float'>
print(type(c))    # <class 'complex'>
print(type(d))    # <class 'str'>
print(type(e))    # <class 'bool'>
```

#### 强制类型转换

- `int()`：将一个数值或字符串（**表示整数的字符串：只包含数字字符的字符串**）转换成整数，可以指定进制。
- `float()`：将一个字符串转换成浮点数。
- `str()`：将指定的对象转换成字符串形式，可以指定编码。
- `chr()`：将整数转换成该编码对应的字符串（一个字符）。
- `ord()`：将字符串（一个字符）转换成对应的编码（整数）。

#### 运算符
| 运算符                                                       | 描述                           |
| ------------------------------------------------------------ | ------------------------------ |
| `[]` `[:]`                                                   | 下标，切片                     |
| `**`                                                         | 指数                           |
| `~` `+` `-`                                                  | 按位取反, 正负号               |
| `*` `/` `%` `//`                                             | 乘，除，模，整除               |
| `+` `-`                                                      | 加，减                         |
| `>>` `<<`                                                    | 右移，左移                     |
| `&`                                                          | 按位与                         |
| `^` `\|`                                                      | 按位异或，按位或               |
| `<=` `<` `>` `>=`                                            | 小于等于，小于，大于，大于等于 |
| `==` `!=`                                                    | 等于，不等于                   |
| `is`  `is not`                                               | 身份运算符                     |
| `in` `not in`                                                | 成员运算符                     |
| `not` `or` `and`                                             | 逻辑运算符                     |
| `=` `+=` `-=` `*=` `/=` `%=` `//=` `**=` `&=` `|=` `^=` `>>=` `<<=` | （复合）赋值运算符             |
		
	
> 打印的方法：1. print('%.1f华氏度 = %.1f摄氏度' **% (f, c)**) 2. print(**f**'{f:.1f}华氏度 = {c:.1f}摄氏度')

### 分支结构

#### 错题：输入三条边长，如果能构成三角形就计算周长和面积。

```Python
"""
判断输入的边长能否构成三角形，如果能则计算出三角形的周长和面积

"""
a = float(input('a = '))
b = float(input('b = '))
c = float(input('c = '))
if a + b > c and a + c > b and b + c > a:
    print('周长: %f' % (a + b + c))
    p = (a + b + c) / 2
    area = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    print('面积: %f' % (area))
else:
    print('不能构成三角形')
```
>  数学基础 ：构成三角形的条件是三角形的任意两边之和大于第三边。<br>
> **说明：** 上面使用的通过边长计算三角形面积的公式叫做[海伦公式](https://zh.wikipedia.org/zh-hans/海伦公式)。

### 循环结构
#### for in 循环 

`range`的用法非常灵活，下面给出了一个例子：

- `range(101)`：可以用来产生0到100范围的整数，取不到101。
- `range(1, 101)`：可以用来产生1到100范围的整数，前闭后开。
- `range(1, 101, 2)`：可以用来产生1到100的奇数，其中2是步长。
- `range(100, 0, -2)`：可以用来产生100到1的偶数，其中-2是步长。

知道了这一点，我们可以用下面的代码来实现1~100之间的偶数求和。
计算1~100求和的结果（$\displaystyle \sum \limits_{n=1}^{100}n$）

#### while 循环

小游戏：计算机出一个1到100之间的随机数，玩家输入自己猜的数字，计算机给出对应的提示信息（大一点、小一点或猜对了），如果玩家猜中了数字，计算机提示用户一共猜了多少次，游戏结束，否则游戏继续。

```Python
import random

ans = random.randint(1, 100)
cnt = 0
while True :
    cnt += 1
    guess = int(input('enter your number: '))
    if guess > ans :
        print('be smaller')
    elif guess < ans :
        print('be bigger')
    else:
        print('Bingo!Times:%d' %cnt)
        break
```

九九乘法表： <br>

- 知道边界，因此优先考虑 `for-in` 循环


```Python
for row in range(1, 10) :
    for column in range(1, row + 1) :
        print('%d ' % (row * column), end = '\t') # 默认end = '\n'
    print('\n')
```

输入两个正整数，计算它们的最大公约数和最小公倍数。

``` python
x = int(input('x='))
y = int(input('y='))

if x > y :
	x, y = y, x

for factor in range(x, 0, -1) : # 递减思想
	if x % factor == 0 and y % factor == 0 :
		print("....%d" % factor)
		print("....%d" % x * y // factor) # 地板除法 
		break
```
打印：
```
    *
   **
  ***
 ****
*****
————————————
    *
   ***
  *****
 *******
*********
```
答案：
```python
for i in range(5):
    for j in range(5):
        if j < 5 - i - 1:
# 用j来控制 先用循环结构来控制层数 再用分支结构来实现具体的打印
            print(' ', end='')
        else:
            print('*', end='')
    print()

# 怎么先打空格在打星星在打空格
for i in range(5):
    for j in range(10):
        if(j < 5 - i - 1):
            print(' ', end='')
        # 对称性！
        elif(j > 5 + i - 1):
            print(' ',end='')
        else:
            print('*',end='')
    print()

# 第二种看法：打完空格打星星 的空格不需要再打印了！
for i in range(5):
    for j in range(5 - i - 1):
        print(' ',end='')
    for j in range(2 * i + 1):
        print('*',end='')
    print()
```
### 构造程序逻辑

1. 实现将一个正整数反转。例如：将12345变成54321
   
- python里：//地板除法， /除出来会是小数，%取模
- % 取模从`低位`分离 然后用`//更新`被除数 在循环%
```python 

# 不确定几位数肿么办 --> 不需要知道 当num==0的时候就分离完所有的位数了

reversed_num = 0
while num > 0:
    reversed_num = reversed_num * 10 + num % 10
    num = num // 10
print('reverse:%d' % reversed_num)

"""
12345 12345%10 = 1234...5;    rn=0+5=5;           num = 1234
1234 1234%10 = 123...4;       rn=50+4=54;         num = 123
123 123%10 = 12...3;          rn=540+3=543;       num = 12
12 12%10 = 1...2;             rn=5430+2=5432;     num = 1
1 1%10 = 0...1;               rn=54320+1=544321   num = 0

EXIT
"""
```




