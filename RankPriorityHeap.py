import random
import numpy as np
from math import log
import math


class MaxHeapObject:
  def __init__(self, score, index, otherIndex):
    self.index = index
    self.score = score
    self.otherIndex = otherIndex
  def __lt__(self, other):
    return self.score > other.score

class RankPriorityHeap:
  def __init__(self, capacity, num_partitions):
    self.num_partitions = num_partitions
    self.heap = []
    self.array = []
    self.capacity = capacity
    self.partitions = []
    self.lastPartitionParameters = None

  def resetPartitions(self, alpha=1, skip=0):
    #skip = 0
    size = len(self.array)

    parameters = (size, alpha, skip)
    if self.lastPartitionParameters == parameters:
      return
    else:
      self.partitions = []
      self.lastPartitionParameters = parameters
      #total = H(size + skip) - H(skip)
      count = np.float32(0.0)
      probs = np.zeros(size, dtype=np.float32)

      for i in range(size):
        count += np.float32(1.0)/pow(1+i+skip, alpha)
        #print(count)
        probs[i] = count

      #print(count)
      indexes = np.searchsorted(probs, count * np.arange(1,self.num_partitions+1) / self.num_partitions, side='right')
      last = 0
      for i in indexes:
        self.partitions.append((last, i))
        last = i
      #print(self.partitions)
      assert(len(self.partitions) == self.num_partitions)
      self.partitions[-1] = (self.partitions[-1][0], -1)
    
  def getRandom(self):
    both = []
    #print(self.partitions)
    for start,end in self.partitions:
      if (end == -1):
        end = len(self.array)
      index = random.randrange(start,end)
      both.append((self.heap[index].otherIndex, float(1.0)/(len(self.partitions)*(end-start))))

    return both



  def add(self, score):
    size = len(self.array)
    heapObject = MaxHeapObject(score, size, size)
    self.array.append(heapObject)
    self.heap.append(heapObject)
    self.heapFixUp(size)
    assert(len(self.array) <= self.capacity)


  def get(self, arrayIndex):
    return self.array[arrayIndex].score

  def size(self):
    return len(self.array)

  def sortHeap(self):
    self.heap.sort()
    for index, x in enumerate(self.heap):
      x.index = index

  def replaceArray(self, arrayIndex, score):
    heapObject = self.array[arrayIndex]
    heapIndex = heapObject.index
    self.heapReplace(heapIndex, score)

  def heapReplace(self, index, replacementScore):
    self.heap[index].score = replacementScore
    self.heapFix(index)

  def heapFix(self, index):
    if index==0:
      self.heapFixDown(index)
    elif self.heap[(index+1)//2 - 1] > self.heap[index]:
      self.heapFixUp(index)
    else:
      self.heapFixDown(index)
    
  def heapFixDown(self, index):
    pos1 = index*2 + 1
    pos2 = index*2 + 2
    while pos1 < self.size():
      if pos2>=self.size():
        child = pos1
      elif self.heap[pos1] < self.heap[pos2]:
        child = pos1
      else:
        child = pos2

      if self.heap[child] < self.heap[index]:
        self.heap[child], self.heap[index] = self.heap[index], self.heap[child]
        self.heap[child].index = child
        self.heap[index].index = index
        index = child
        pos1 = index*2 + 1
        pos2 = index*2 + 2
      else:
        break

  def heapFixUp(self, index):
    while index > 0:
      prev = (index+1)//2 - 1 
      if self.heap[prev] > self.heap[index]:
        self.heap[prev], self.heap[index] = self.heap[index], self.heap[prev]
        self.heap[prev].index = prev
        self.heap[index].index = index
        index = prev
      else:
        break