# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
#
# import numpy as np
# from matplotlib import pyplot as plt
#
# x = np.arange(0, 6, 0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # 绘制图形
# plt.plot(x, y1, label='sin')
# plt.plot(x, y2, linestyle='--', label='cos')
# plt.xlabel('时间', fontproperties='SimHei')     # fontproperties='SimHei' 解决中文显示问题
# plt.ylabel('努力程度', fontproperties='SimHei')
# plt.title('今日努力报告', fontproperties='SimHei')
# plt.legend()
# plt.show()
#
#
class Person:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        return "Hello, my name is " + self.name

# 创建一个 Person 对象
person = Person("Alice")

# 使用 . 来访问 name 属性
print(person.name)  # 输出: Alice

# 使用 . 来调用 say_hello 方法
print(person.say_hello())  # 输出: Hello, my name is Alice
