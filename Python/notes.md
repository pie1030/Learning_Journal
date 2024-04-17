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

```python
a = 0
b = 1
for _ in range(20):
    a, b = b, a + b
    print(a, end=' ')
```
- elegant code in python ~

### 函数和模块的使用
#### 函数的参数

在Python中，函数的参数可以有默认值，也支持使用可变参数

- 默认值
```python
def roll_dice(n=2):
    """摇色子"""
    total = 0
    for _ in range(n):
        total += randint(1, 6)
    return total

# 如果没有指定参数那么使用默认值摇两颗色子
print(roll_dice())
# 摇三颗色子
print(roll_dice(3))
```
- 可变参数(具体有多少个参数是由调用者来决定。不确定参数个数时：)
```python
# 在参数名前面的*表示args是一个可变参数
def add(*args):
    total = 0
    for val in args:
        total += val
    return total


# 在调用add函数时可以传入0个或多个参数
print(add())
print(add(1))
print(add(1, 2))
print(add(1, 2, 3))
print(add(1, 3, 5, 7, 9))
```

**语法上的一小步 思维上的一大步！**
```python
def main():
    # Todo: Add your code here
    pass


if __name__ == '__main__':
    main()
```

### 字符串和常用数据结构
#### 使用字符串
Q:为什么要学字符串！
A:因为今天的计算机处理得更多的数据可能都是以文本的方式存在的qaq<br><br>
单个或多个字符用`单引号`或者`双引号`包围起来，就可以表示一个字符串。<br><br>

- 在字符串中使用\（反斜杠）来表示转义，也就是说\后面的字符不再是它原来的意义
    - `\`后面还可以跟一个`八进制`或者`十六进制数`来表示字符
    - `\`后面跟`Unicode字符`编码来表示字符
```python
s1 = '\141\142\143\x61\x62\x63'
# 1 + 32 + 64 = 97； 1 + 16 * 6 = 97
s2 = '\u57f9\u5b50' # 运行有惊喜qwqqq
print(s1, s2)
```
    - 通过在字符串的最前面加上字母r来取消`\`的转义
```python
s1 = r'\'hello, world!\''
s2 = '\n\\hello, world!\\\n'
print(s1, s2, end='')
```
#### 丰富的运算符
1. `+` ：字符串连接
2. `*` : 字符串重复
3. `in` and `not in` : 是否包含
4. `[]` and `[:]` : 切片运算

### 字符串类型 VS 数值类型
数值类型是`标量`类型，也就是说这种类型的对象没有可以访问的内部结构；而字符串类型是一种`结构化的、非标量`类型，所以才会有一系列的属性和方法。

### 列表
下面的代码演示了如何定义列表、如何遍历列表以及列表的下标运算。
```
list1 = [1, 3, 5, 7, 100]
print(list1) # [1, 3, 5, 7, 100]
# 乘号表示列表元素的重复
list2 = ['hello'] * 3
print(list2) # ['hello', 'hello', 'hello']
# 计算列表长度(元素个数)
print(len(list1)) # 5
# 下标(索引)运算
print(list1[0]) # 1

# print(list1[5])  # IndexError: list index out of range

print(list1[-1]) # 100

list1[2] = 300
print(list1) # [1, 3, 300, 7, 100]

# 通过循环用下标遍历列表元素
for index in range(len(list1)):
    print(list1[index])

# 通过for循环遍历列表元素
for elem in list1:
    print(elem)

# 通过enumerate函数处理列表之后再遍历可以同时获得元素索引和值
for index, elem in enumerate(list1):
    print(index, elem)
```
### 生成式和生成器
生成器对象是一种特殊的迭代器，可以用于生成一系列的值。
`生成器对象`可以通过函数中包含`yield`关键字来`创建`。当函数中包含yield关键字时，该函数就成为一个`生成器函数`，调用生成器函数会返回一个`生成器对象`。

生成器对象可以用于`按需`生成值，而不是一次性生成所有值并保存在内存中。每次调用生成器对象的next()方法或使用for循环来迭代时，生成器函数会从上次暂停的地方继续执行，直到遇到下一个yield语句或函数结束。

生成器对象的一个常见用途是在处理大量数据时`节省内存`，因为它们可以按需生成数据而不会一次性占用大量内存。

```python
# 定义生成器函数
def my_generator(n):
    for i in range(n):
        yield i

# 创建生成器对象
gen = my_generator(5)

# 遍历生成器对象
for value in gen:
    print(value)
```
`my_generator`是一个生成器函数，它使用`yield`关键字来生成`值`。调用my_generator(5)会返回一个`生成器对象gen`，该生成器对象可以用于按需生成0到4这5个值。通过for循环遍历生成器对象时，会依次打印出生成的值。

```python
"""
还是斐波那契哈哈
"""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
        yield a


def main():
    for val in fib(20):
        print(val)


if __name__ == '__main__':
    main()
```

### 元组
列表类似也是一种容器数据类型，可以用一个变量（对象）来存储多个数据，不同之处在于元组的元素**不能修改**。(TypeError: 'tuple' object does not support item assignment)
> 顾名思义，我们把多个元素组合到一起就形成了一个元组，所以它和列表一样可以保存多条数据。


我们已经有了列表这种数据结构，为什么还需要元组这样的类型呢？

元组中的元素是**无法修改**的，事实上我们在项目中尤其是`多线程环境`中可能更喜欢使用的是那些不变对象（一方面因为对象状态不能修改，所以可以避免由此引起的不必要的程序错误，容易维护；
另一方面因为没有任何一个线程能够修改不变对象的`内部状态`，一个不变对象自动就是`线程安全`的，这样就可以**省掉处理同步化的开销**。
一个不变对象可以方便的被**共享访问**）。

元组在**创建时间和占用的空间**上面都优于列表。
我们可以使用sys模块的getsizeof函数来检查存储同样的元素的元组和列表各自占用了多少内存空间。

### 集合
可以按照下面代码所示的方式来创建和使用集合。
```python
# 创建集合的字面量语法
set1 = {1, 2, 3, 3, 3, 2}
print(set1)
print('Length =', len(set1))
# 创建集合的构造器语法(面向对象部分会进行详细讲解)
set2 = set(range(1, 10))
set3 = set((1, 2, 3, 3, 2, 1))
print(set2, set3)
# 创建集合的推导式语法(推导式也可以用于推导集合)
set4 = {num for num in range(1, 100) if num % 3 == 0 or num % 5 == 0}
print(set4)
```

集合的交集、并集、差集、对称差运算
```python
print(set1 & set2)
# print(set1.intersection(set2))
print(set1 | set2)
# print(set1.union(set2))
print(set1 - set2)
# print(set1.difference(set2))
print(set1 ^ set2)
# print(set1.symmetric_difference(set2))
# 判断子集和超集
print(set2 <= set1)
# print(set2.issubset(set1))
print(set3 <= set1)
# print(set3.issubset(set1))
print(set1 >= set2)
# print(set1.issuperset(set2))
print(set1 >= set3)
# print(set1.issuperset(set3))
```

### 字典
字典是另一种`可变`容器模型，它可以存储`任意类型`对象，与列表、集合不同的是，字典的每个元素都是由一个键和一个值组成的“`键值对`”，键和值通过冒号分开。
```python
# 创建字典的字面量语法
scores = {'骆昊': 95, '白元芳': 78, '狄仁杰': 82}
print(scores)
# 创建字典的构造器语法
items1 = dict(one=1, two=2, three=3, four=4)
# 通过zip函数将两个序列压成字典
items2 = dict(zip(['a', 'b', 'c'], '123'))
# 创建字典的推导式语法
items3 = {num: num ** 2 for num in range(1, 10)}
print(items1, items2, items3)
# 通过键可以获取字典中对应的值
print(scores['骆昊'])
print(scores['狄仁杰'])
# 对字典中所有键值对进行遍历
for key in scores:
    print(f'{key}: {scores[key]}')
# 更新字典中的元素
scores['白元芳'] = 65
scores['诸葛王朗'] = 71
scores.update(冷面=67, 方启鹤=85)
print(scores)
if '武则天' in scores:
    print(scores['武则天'])
print(scores.get('武则天'))
# get方法也是通过键获取对应的值但是可以设置默认值
print(scores.get('武则天', 60))
# 删除字典中的元素
print(scores.popitem())
print(scores.popitem())
print(scores.pop('骆昊', 100))
# 清空字典
scores.clear()
print(scores)
```
