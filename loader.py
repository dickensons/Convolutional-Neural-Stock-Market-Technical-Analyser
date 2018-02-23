import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base
import math
import numpy as np
import pandas as pd
import functools as ft
import csv
import os
import re


np.set_printoptions(threshold=np.nan)

class DataSet(object):
  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               seed=None):

    self.check_data(images, labels)
    seed1, seed2 = random_seed.get_seed(seed)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._total_batches = images.shape[0]

  def check_data(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def total_batches(self):
    return self._total_batches

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    start = self._index_in_epoch

    # first epoch shuffle
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._total_batches)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]

    # next epoch
    if start + batch_size <= self._total_batches:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

    # if the epoch is ending
    else:
      self._epochs_completed += 1

      # store what is left of this epoch
      batches_left = self._total_batches - start
      images_left = self._images[start:self._total_batches]
      labels_left = self._labels[start:self._total_batches]

      # shuffle for new epoch
      if shuffle:
        perm = np.arange(self._total_batches)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]

      # start next epoch
      start = 0
      self._index_in_epoch = batch_size - batches_left
      end = self._index_in_epoch
      images_new = self._images[start:end]
      labels_new = self._labels[start:end]
      return np.concatenate((images_left, images_new), axis=0), np.concatenate((labels_left, labels_new), axis=0)

def load_csv(fname, col_start=1, row_start=1, delimiter=",", dtype=dtypes.float32):
  data = np.genfromtxt(fname, delimiter=delimiter)
  for _ in range(col_start):
    data = np.delete(data, (0), axis=1)
  for _ in range(row_start):
    data = np.delete(data, (0), axis=0)
  # remove two unnecessary columns
  data = data[:,:5]
  # use less data for test
  data = data[1000:5000]
  print('data.shape ', data.shape)
  # print(np.transpose(data))
  return data

# Helper to order files
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# stock data loading
def load_stock_data(path, moving_window=128, columns=5, train_test_ratio=4.0):
  # process a single file's data into usable arrays
  # def process_data(data):
  #   stock_set = np.zeros([0,2,moving_window,columns])
  #   label_set = np.zeros([0,2])
  #   # start from 1152 because we need to add 128 prices for each 10 minutes in the past
  #   for idx in range(1152,data.shape[0] - (moving_window + 5)):
  #     min_data   = data[range(idx,idx+(moving_window)),:]
  #     e10_min_data = data[range((idx+moving_window)-(moving_window*10),idx+(moving_window),10)]
  #     temp = []
  #     temp.append(min_data)
  #     temp.append(e10_min_data)
  #     stock_set = np.concatenate((stock_set, np.expand_dims(temp, axis=0)), axis=0)
  #     if idx % 500 == 0:
  #       print("index: " , idx)
  #     #if data[idx+(moving_window+5),3] > data[idx+(moving_window),3]:
  #     # true if the price will rise at least 0.1 %
  #     average_price_in_5_min = np.average(data[range((idx + (moving_window)),((idx + (moving_window + 5)))),3])
  #     if average_price_in_5_min > (data[idx + (moving_window), 3]*1.00001):
  #       lbl = [[1.0, 0.0]]
  #     else:
  #       lbl = [[0.0, 1.0]]
  #     label_set = np.concatenate((label_set, lbl), axis=0)
  #     # label_set = np.concatenate((label_set, np.array([data[idx+(moving_window+5),3] - data[idx+(moving_window),3]])))
  #   print(stock_set.shape, label_set.shape)
  #   return stock_set, label_set

  # read a directory of data

  train_stocks = np.load("input/train_stocks.npy")
  train_labels = np.load("input/train_labels.npy")

  test_stocks = np.load("input/test_stocks.npy")
  test_labels = np.load("input/test_labels.npy")

  train = DataSet(train_stocks, train_labels)
  test = DataSet(test_stocks, test_labels)

  return base.Datasets(train=train, validation=None, test=test)

# db = load_stock_data("data/short/")
# images, labels = db.train.next_batch(10)
# print(images.shape, labels.shape)
# print(images, labels)
