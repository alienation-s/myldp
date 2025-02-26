# 学员：南格格
# 时间：2022/8/14 10:04
import math
import random
import mmh3
import numpy as np
from bitarray import bitarray
import os.path
import re


# bitarray长度
BIT_SIZE = 100
epsilon=0.5
class BloomFilter():

  def __init__(self):
    bit_array = bitarray(BIT_SIZE)
    bit_array.setall(0)
    self.bit_array = bit_array
    self.bit_size = self.length()
    self.bit_final_array=bit_array
    self.epsilon=epsilon


  def get_points(self, url):
    """
    生成需要插入的位置
    :param url:
    :return:节点的列表
    """
    point_list = []
    for i in range(7):
      point = mmh3.hash(url,30+i) % self.bit_size
      # print(point)
      # print("point")
      point_list.append(point)
    return point_list

  def add(self, url):
    """
    添加url到bitarray中
    :param url:
    :return:
    """
    res = self.bitarray_expand()
    points = self.get_points(url)
    try:
      for point in points:
        self.bit_array[point] = 1
      return '注册完成！'
    except Exception as e:
      return e

  def contains(self,url):
    """
    验证url是否存在
    :param url:
    :return:True or False
    """
    points = self.get_points(url)
    # 在bitarray中查找对应的点，如果有一个点值为0就说明该url不存在
    for p in points:
      if self.bit_array[p] == 0:
        return False
    return True


  def count(self):
    """
    获取bitarrray中使用的节点数
    :return: bitarray长度
    """
    return self.bit_array.count()


  def length(self):
    """
    获取bitarray的长度
    :return:bitarray的长度
    """
    print(self.bit_array)
    return len(self.bit_array)


  def bitarray_expand(self):
    """
    扩充bitarray长度
    :return:bitarray的长度或使用率，布隆过滤器的bitarray的使用最好不要超过50%,这样误判率低一些
    """
    isusespace = round(int(self.count()) / int(self.length()),4)
    if 0.50 < isusespace:
      # 新建bitarray
      expand_bitarray = bitarray(BIT_SIZE)
      expand_bitarray.setall(0)
      # 增加新建的bitarray
      self.bit_array = self.bit_array + expand_bitarray
      self.bit_size = self.length()
      return self.bit_size
    else:
      return f'长度尚可,{round(isusespace * 100,2)}%'



  def get_captcha():
    """
    生成用于测试的随机码
    :return:
    """
    seed = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    captcha = ""
    for i in range(2):
      captcha += random.choice(seed)
    print(captcha)
    return captcha

  def randomData(self):
    for i in range(self.length()):
      p = math.exp(epsilon) / (1 + math.exp(epsilon))
      if np.random.rand() <= p:
        self.bit_final_array[i] = np.random.binomial(n= 1, p= 0.5, size= 1)[0]
      else:
        self.bit_final_array[i] = self.bit_array[i]
    return self.bit_final_array


if __name__ == '__main__':
  bloom = BloomFilter()
  # for i in range(2):
  #   # bloom.add(f'www.{get_captcha()}.com')
  #   print(bloom.length())
  #   print(bloom.count())
  bloom.add('50')
  print(bloom.bit_array)
  # print(bloom.length())
  print(bloom.count())
  print(bloom.randomData())
  # 获取到值
  # print('88888888')
  # for i in range(100):
  #   if(bloom.contains(str(i))):
  #     print(i)

  # print(bloom.contains('12'))

