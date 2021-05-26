#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: eth_predict
# Created on: 2021/4/18

import os, time, psutil
import matplotlib.pyplot as plt
import random
import pathlib
import json

import numpy
import pandas as pd
import numpy as np
from api.market_db import *
from i2cylib.utils.path.path_fixer import *
from i2cylib.utils.stdout.echo import *

# 不使用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# *屏蔽tensorflow警告信息输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# *RTX硬件兼容性修改配置
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

DATASET_INDEX = "database/index.npz"  # 数据集预处理索引ID
TARGETMARKET = "ETHUSDT"

TEST_RATE = 0.05  # 测试集比例
BATCH_SIZE = 128  # 批处理大小
SAMPLE_SIZE = 720  # 特征样本大小
PREDICT_SIZE = 80  # 预测输出大小
SAMPLE_TIME_MS = 7500  # 采样间隔时间（单位ms）
EPOCHES = 100  # 训练代数
BUFF_RATE = 0.1  # 缓冲区大小指数
LEARNING_RATE = 0.0001  # 学习率
MODEL_FILE = "rnn/models/eth_market_model.h5"  # 在此处修改神经网络模型文件
NAME = "CryptoCoinPrediction"

ETH_RNN = None
DATASET = None


class LSTMLayer4(tf.keras.layers.Layer):

    def __init__(self, units):
        super(LSTMLayer4, self).__init__()
        #self.bn_1 = tf.keras.layers.BatchNormalization()
        self.units = units
        self.lstm_1 = None
        self.lstm_2 = None
        self.lstm_3 = None
        self.lstm_4 = None

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.lstm_4(x)
        return x

    def build(self, input_shape):
        self.lstm_1 = tf.keras.layers.LSTM(self.units, input_shape=input_shape, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(self.units * 2, return_sequences=True, dropout=0.05)
        self.lstm_3 = tf.keras.layers.LSTM(self.units * 2, return_sequences=True, dropout=0.1)
        self.lstm_4 = tf.keras.layers.LSTM(self.units * 4, dropout=0.1)


class Conv1DLSTM4(tf.keras.layers.Layer):

    def __init__(self, units, core):
        super(Conv1DLSTM4, self).__init__()
        self.conv1d_1 = tf.keras.layers.Conv1D(units, core, padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv1d_2 = tf.keras.layers.Conv1D(units // 2, core, padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.lstm_1 = tf.keras.layers.LSTM(units // 2, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(units // 2, return_sequences=True, dropout=0.05)
        self.lstm_3 = tf.keras.layers.LSTM(units // 2, dropout=0.1)

    def call(self, inputs):
        x = self.conv1d_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv1d_2(inputs)
        x = self.bn_2(x)
        x = tf.nn.relu(x)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        return x


class customNN:
    def __init__(self, model_name="MLP"):
        self.name = model_name
        self.train_db = None
        self.test_db = None
        self.model = None
        self.train_size = 0
        self.test_size = 0
        self.data_shape = []
        self.batch_size = 8
        self.train_history = None
        self.tensorboard_enable = False
        self.log_root = "./tensorflow_log"
        self.callbacks = []
        self.callback_file_writer = None
        self.base_model = None
        self.epoch = 0
        self.model_file = "{}.h5".format(self.name)
        self.autosave = False
        self.output_counts = 0
        self.finetune_level = 0

        self.call_times = 0

    def _get_freeRAM(self):
        free_ram = psutil.virtual_memory().free
        return free_ram

    def _init_tensorboard(self):
        log_dir = os.path.join(self.log_root,
                               time.strftime("%Y%m%d-%H%M%S_") +
                               self.name
                               )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,
                                                              histogram_freq=1)
        self.callbacks.append(tensorboard_callback)
        self.callback_file_writer = tf.summary.create_file_writer(os.path.join(
            log_dir, "train"))
        self.callback_file_writer.set_as_default()

    def load_dataset(self, trainset, testset=None,
                     mapFunc=None, testRate=0.15, batchSize=8,
                     shufflePercentage=0.3, mapFuncTest=None,
                     mapFuncLabel=None, mapFuncLabelTest=None):  # dataset has to be formated tensors: (data, labels)
        self.batch_size = batchSize
        if testset == None:
            # randomly split trainset and testset
            datasets = [ele for ele in trainset]
            #print(datasets, trainset)
            train_size = len(datasets[0]) - int(len(datasets[0]) * testRate)
            all_indexs = list(range(len(datasets[0])))
            random.shuffle(all_indexs)
            features = []
            labels = []
            if (type(datasets[1][0]) in (type([0]), type((0,)))) and len(datasets[1][0]) == len(all_indexs):
                for i in enumerate(datasets[1]):
                    labels.append([])
                    self.output_counts += 1
                for index in all_indexs[:train_size]:
                    data = datasets[0][index]
                    features.append(data)
                    for i, l in enumerate(datasets[1]):
                        label = datasets[1][i][index]
                        labels[i].append(label)
                if type(labels[0]) == type([0]):
                    labels = tuple(labels)
            else:
                self.output_counts += 1
                for index in all_indexs[:train_size]:
                    features.append(datasets[0][index])
                    labels.append(datasets[1][index])
            trainset = (features, labels)
            features = []
            labels = []
            if (type(datasets[1][0]) in (type([0]), type((0,)))) and len(datasets[1][0]) == len(all_indexs):
                for i in enumerate(datasets[1]):
                    labels.append([])
                for index in all_indexs[train_size:]:
                    data = datasets[0][index]
                    features.append(data)
                    for i, l in enumerate(datasets[1]):
                        label = datasets[1][i][index]
                        labels[i].append(label)
                if type(labels[0]) == type([0]):
                    labels = tuple(labels)
            else:
                for index in all_indexs[train_size:]:
                    features.append(datasets[0][index])
                    labels.append(datasets[1][index])
            testset = (features, labels)

        self.data_shape = tf.constant(trainset[0][0]).shape
        self.train_size = len(trainset[0])
        self.test_size = len(testset[0])

        print("trainset sample number: {}".format(str(self.train_size)))
        print("testset sample number: {}".format(str(self.test_size)))

        if mapFunc == None:
            if mapFuncLabel == None:
                train_db = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(trainset[0]),
                                                tf.data.Dataset.from_tensor_slices(trainset[1])))
                test_db = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(testset[0]),
                                               tf.data.Dataset.from_tensor_slices(testset[1])))
            else:
                if mapFuncLabelTest == None:
                    mapFuncLabelTest = mapFuncLabel
                train_db = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(trainset[0]), tf.data.Dataset.from_tensor_slices(
                        trainset[1]).map(mapFuncLabel, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

                test_db = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(testset[0]), tf.data.Dataset.from_tensor_slices(
                        testset[1]).map(mapFuncLabelTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

        else:
            if mapFuncTest == None:
                mapFuncTest = mapFunc
            self.data_shape = mapFunc(trainset[0][0]).shape
            train_db = tf.data.Dataset.from_tensor_slices(trainset[0])
            train_db = train_db.map(mapFunc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_db = tf.data.Dataset.from_tensor_slices(testset[0])
            test_db = test_db.map(mapFuncTest)

            if mapFuncLabel == None:
                train_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(trainset[1])))
                test_db = tf.data.Dataset.zip((
                    test_db, tf.data.Dataset.from_tensor_slices(testset[1])))
            else:
                if mapFuncLabelTest == None:
                    mapFuncLabelTest = mapFuncLabel
                train_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(
                        trainset[1]).map(mapFuncLabel, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

                test_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(
                        testset[1]).map(mapFuncLabelTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

        datasize = 1
        for size in self.data_shape:
            datasize *= size
        freeRAM = int(self._get_freeRAM() * shufflePercentage)
        print("free RAM size: {} MB".format(str(freeRAM // 1048576)))

        shuffle_MaxbuffSize = int((freeRAM * 0.8) // datasize)
        prefetch_buffSize = int((freeRAM * 0.2) // (datasize * self.batch_size))

        print("automatically allocated data buffer size: {} MB".format(str(shuffle_MaxbuffSize * datasize // 1048576)))

        shuffle_buffSize = shuffle_MaxbuffSize
        if shuffle_MaxbuffSize > self.train_size:
            shuffle_buffSize = self.train_size
        train_db = train_db.shuffle(shuffle_buffSize).repeat().batch(self.batch_size).prefetch(prefetch_buffSize)
        shuffle_buffSize = shuffle_MaxbuffSize
        if shuffle_MaxbuffSize > self.test_size:
            shuffle_buffSize = self.test_size
        test_db = test_db.shuffle(shuffle_buffSize).repeat().batch(self.batch_size).prefetch(prefetch_buffSize)

        self.train_db = train_db
        self.test_db = test_db

    def set_model_file(self, path):
        self.model_file = path

    def enable_tensorboard(self, log_dir_root="./tensorflow_log"):
        self.log_root = log_dir_root
        self.tensorboard_enable = True

    def enable_checkpointAutosave(self, path=None):
        if path != None:
            self.model_file = path
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_file)
        self.add_callback(checkpoint)
        self.autosave = True

    def add_callback(self, callback_func):  # all callbacks added will be reset after training
        self.callbacks.append(callback_func)

    def init_model(self):  # 神经网络模型

        input_1 = tf.keras.Input(shape=(SAMPLE_SIZE, 11*5), name="input_1")
        input_A = tf.keras.Input(shape=(SAMPLE_SIZE, 300), name="input_A")
        input_B = tf.keras.Input(shape=(SAMPLE_SIZE, 40), name="input_B")

        X1 = input_1
        XA = input_A
        XB = input_B

        X1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.05)(X1)
        X1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.1)(X1)
        X1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.1)(X1)
        X1 = tf.keras.layers.LSTM(256, dropout=0.1)(X1)

        XA = tf.keras.layers.Conv1D(128, 7, padding="same")(XA)
        XA = tf.keras.layers.BatchNormalization()(XA)
        XA = tf.nn.relu(XA)
        XA = tf.keras.layers.Conv1D(64, 7, padding="same")(XA)
        XA = tf.keras.layers.BatchNormalization()(XA)
        XA = tf.nn.relu(XA)
        XA = tf.keras.layers.LSTM(64, dropout=0.05, return_sequences=True)(XA)
        XA = tf.keras.layers.LSTM(64, dropout=0.1, return_sequences=True)(XA)
        XA = tf.keras.layers.LSTM(64, dropout=0.1)(XA)

        XB = tf.keras.layers.Conv1D(32, 7, padding="same")(XB)
        XB = tf.keras.layers.BatchNormalization()(XB)
        XB = tf.nn.relu(XB)
        XB = tf.keras.layers.Conv1D(16, 7, padding="same")(XB)
        XB = tf.keras.layers.BatchNormalization()(XB)
        XB = tf.nn.relu(XB)
        XB = tf.keras.layers.LSTM(64, dropout=0.05, return_sequences=True)(XB)
        XB = tf.keras.layers.LSTM(64, dropout=0.1, return_sequences=True)(XB)
        XB = tf.keras.layers.LSTM(64, dropout=0.1)(XB)

        X1 = tf.keras.layers.BatchNormalization()(X1)
        XA = tf.keras.layers.BatchNormalization()(XA)
        XB = tf.keras.layers.BatchNormalization()(XB)

        XD = tf.keras.layers.concatenate([X1, XA, XB])
        XD = tf.keras.layers.Dense(1024, activation="relu")(XD)
        XD = tf.keras.layers.Dense(1024, activation="relu")(XD)
        XD = tf.keras.layers.Dense(512, activation="relu")(XD)
        XD = tf.keras.layers.Dense(PREDICT_SIZE, activation="relu")(XD)

        self.model = tf.keras.Model(inputs=[input_1, input_A, input_B],
                                    outputs=XD)

        self.compile_model()
        print(self.model.summary())

    def postProc_model(self, finetune_level=None):  # 模型后期处理（微调）
        pass

    def compile_model(self):
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('loss', patience=3, factor=0.5, min_lr=0.000001)
        self.callbacks.append(lr_reduce)
        loss = tf.keras.losses.Huber(delta=2.4)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                           loss=loss,  # 均方差预测问题
                           metrics=["mae"]
                           )

    def save_model(self, path=None):
        if path != None:
            self.model_file = path
        self.model.save(self.model_file)

    def load_model(self, path=None):
        if path != None:
            self.model_file = path
        self.model = tf.keras.models.load_model(self.model_file, compile=True)
        self.compile_model()

    def train(self, epochs=100):
        if self.tensorboard_enable and self.epoch == 0:
            self._init_tensorboard()
        try:
            self.train_history = self.model.fit(self.train_db,
                                                epochs=epochs,
                                                initial_epoch=self.epoch,
                                                #steps_per_epoch=self.train_size // self.batch_size,
                                                validation_data=self.test_db,
                                                #validation_steps=self.test_size // self.batch_size,
                                                callbacks=self.callbacks
                                                )
            self.epoch += epochs
        except KeyboardInterrupt:
            print("\ntraining process stopped manually")
            if self.autosave:
                self.load_model(self.model_file)

    def show_history_curves(self):
        plt.plot(self.train_history.epoch, self.train_history.history["loss"], label="Loss_Train")
        plt.plot(self.train_history.epoch, self.train_history.history["acc"], label="Acc_Train")
        plt.plot(self.train_history.epoch, self.train_history.history["val_loss"], label="Loss_Test")
        plt.plot(self.train_history.epoch, self.train_history.history["val_acc"], label="Acc_Test")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(self.name)
        plt.legend()
        plt.show()

    def _generate_bar(self, percent):
        res = " {}%\t[".format(str(round(percent * 100, 1)))
        if percent == 1:
            res += "=" * 30
        else:
            done = int(30 * percent)
            res += done * "="
            res += ">"
            res += (29 - done) * " "

        res += "]"
        return res

    def evaluate(self):
        print("evaluating model with test datasets...")

        acc = self.model.evaluate(self.test_db, return_dict=True,
                                  steps=self.test_size // self.batch_size)

        return acc

    def predict(self, data):
        try:
            if len(data.shape) != len(self.data_shape) + 1:
                data = tf.expand_dims(data, 0)
        except:
            pass
        res = self.model.predict(data)
        self.call_times += 1
        return res


class predictor:

    def __init__(self, dnn):
        self.dnn = dnn
        self.labels_converter = []
        self.pre_process_func = self._default_preprocess

    def _default_preprocess(self, data):
        return data

    def load_labels(self, label_names):  # label_names 必须为列表 [[标签列表1], [标签列表2]]
        for label in label_names:
            converter = dict((index, name)
                             for index, name in enumerate(label))
            self.labels_converter.append(converter)

    def set_preprocess_func(self, func):
        self.pre_process_func = func

    def predict(self, data):
        res_raw = self.dnn.predict(
            self.pre_process_func(data)
        )
        res = []
        for index, converter in enumerate(self.labels_converter):
            res.append(converter.get(tf.argmax(res_raw[index][0]).numpy()))
        return res


class Dataframe(object):

    def __init__(self, data=None, header=None, dtype="int64", no_nan=True):
        if not isinstance(header, list) and header is not None:
            raise Exception("header must be list")
        self.headers = header
        if data is not None:
            if isinstance(data, np.ndarray):
                self.data = data
            else:
                self.data = np.array(data, dtype=dtype)
            if self.headers is None or len(self.headers) < self.data.shape[1]:
                self.headers = list(range(self.data.shape[1]))
        else:
            if self.headers is not None:
                self.data = np.empty((0, len(header)), dtype=dtype)
            else:
                self.data = None
        self.dtype = dtype
        self.offset = 0
        self.no_nan = no_nan

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        if self.offset >= len(self):
            raise StopIteration
        ret = self.data[self.offset]
        self.offset += 1
        return ret

    def __getitem__(self, item):
        try:
            if isinstance(item, tuple):
                index = item[0]
                key = item[1]
                if isinstance(index, dict):
                    index = index[list(index.keys())[0]]
                    if isinstance(index, int) or isinstance(index, slice):
                        pass
                    else:
                        index = 0
                    ret = self.data[int(index)]
                else:
                    if isinstance(index, int) or isinstance(index, slice):
                        pass
                    else:
                        index = 0
                    ret = self.data[index]
                if len(ret.shape) == 1:
                    ret = ret[self.headers.index(key)]
                else:
                    ret = [ele[self.headers.index(key)] for ele in ret]
            else:
                ret = self.data[item]
        except IndexError:
            print("got index {}".format(item))
            raise IndexError
        return ret

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data.__array__())

    def append(self, data):
        if self.headers is None:
            self.headers = list(range(len(data)))
            self.data = np.empty((0, len(self.headers)), dtype=self.dtype)
        raw = [np.nan for ele in self.headers]
        ele_count = 0
        if isinstance(data, dict):
            for ele in data.keys():
                raw[self.headers.index(ele)] = data.get(ele)
                ele_count += 1
        else:
            for i, ele in enumerate(data):
                raw[i] = ele
            ele_count = len(data)
        if ele_count < len(self.headers) and self.no_nan:
            raise KeyError("dataframe width is {}, but the input data has length {}".format(
                len(self.headers),
                len(data.keys())
            ))
        raw = np.array([raw], dtype=self.dtype)
        self.data = np.append(self.data, raw, axis=0)

    def delete(self, **kwargs):
        self.data = np.delete(kwargs)

    def dump(self, filename):
        if len(filename) < 4 or filename[-4:] != ".npz":
            filename += ".npz"
        np.savez_compressed(filename, data=self.data, headers=np.array(self.headers))

    def load(self, filename):
        if len(filename) < 4 or filename[-4:] != ".npz":
            filename += ".npz"
        try:
            raw = np.load(filename)
            self.data = raw["data"]
        except ValueError:
            raw = np.load(filename, allow_pickle=True)
            self.data = raw["data"]
        self.headers = raw["headers"].tolist()

    def head(self, index):
        ret = ""
        for ele in self.headers:
            ret += "{}\t".format(ele)
        ret += "\n"
        for i in range(index):
            ret += str(self.data[i]) + "\n"
        return ret


class DatasetBase:

    def __init__(self, market_db_api, sample_time_ms=3000, set_size=SAMPLE_SIZE,
                 echo=None, index_json=DATASET_INDEX, use_index_json=True,
                 label_size=PREDICT_SIZE):
        if not isinstance(market_db_api, MarketDB):
            raise TypeError("market database API must be a MarketDB object")
        if not isinstance(echo, Echo):
            echo = self.__internal_echo__()
        self.indexed = use_index_json
        if not os.path.exists(index_json):
            self.indexed = False
        self.index_file = index_json
        self.echo = echo
        self.db_api = market_db_api
        self.sample_time = sample_time_ms
        self.set_size = set_size
        self.label_size = label_size
        self.headers = ["TS"] + [ele for ele in self.db_api.monitoring]
        self.index_batches = Dataframe(header=self.headers, dtype=object)
        self.index_valids = Dataframe(header=["TS", "OFFSET"])
        if self.indexed:
            self.read_index()

    class __internal_echo__:

        def __init__(self):
            pass

        def print(self, msg):
            pass

        def buttom_print(self, msg):
            pass

    def __len__(self):
        return len(self.index_valids)

    def __fetch__(self, start_ts):
        ret = {}
        for i in self.db_api.monitoring:
            ret.update({i: []})
        data = self.db_api.fetch(start_ts, start_ts + self.sample_time)
        for i in data.keys():
            if not data[i]["market"]:
                ret = None
                break
            else:
                ret[i].append([ele[0] for ele in data[i]["market"]])
        return ret

    def init_dataset(self):
        if self.indexed:
            return
        start_ts = 0
        end_ts = time.time() * 1000
        for dbn in self.db_api.all_tables.keys():
            db = self.db_api.all_tables[dbn]
            sts = db[0][1]
            ets = db[-1][1]
            if sts > start_ts:
                start_ts = sts
            if end_ts > ets:
                end_ts = ets
        self.echo.print("[dataset] fetching data between {} to {}".format(
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(start_ts / 1000)),
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(end_ts / 1000))
        ))
        offset = start_ts
        last_ids = {}
        lengths = {}
        data = {}
        for ele in self.db_api.monitoring:
            last_ids.update({ele: 1})
            lengths.update({ele: len(self.db_api.all_tables[ele])})
            self.echo.print("[dataset] [{}] length: {}".format(ele, lengths[ele]))
        tick = 0
        while offset <= end_ts:
            try:
                if tick % 100 == 0:
                    self.echo.buttom_print("[dataset] processing data at {}".format(time.strftime(
                        "%y-%m-%d %H:%M:%S", time.localtime(offset / 1000))))
                tick += 1
                data = {}
                for ele in self.db_api.monitoring:
                    data.update({ele: []})
                    while True:
                        if last_ids[ele] > lengths[ele]:
                            break
                        ts_raw = self.db_api.all_tables[ele].get(last_ids[ele], column_name="TIMESTAMP")
                        if ts_raw[0][0] < (offset + self.sample_time):
                            # print((offset + self.sample_time), ts_raw[0][0])
                            data[ele].append(last_ids[ele])
                        else:
                            break
                        last_ids[ele] += 1
                    if not data[ele]:
                        data = None
                        break
            except KeyboardInterrupt:
                self.echo.print("[dataset] process interrupted by keyboard")
            # print("time cost {}s, offset now: {}".format(time.time()-t, time.strftime("%y-%m-%d %H:%M:%S", time.localtime(offset / 1000))))
            if data is None:
                # print(data)
                offset += self.sample_time
                continue
            else:
                data.update({"TS": offset})
                self.index_batches.append(data)
            offset += self.sample_time

        self.echo.print("[dataset] \n{}".format(self.index_batches.head(5)))
        self.echo.buttom_print("")

        self.echo.print("[dataset] proceed {} samples".format(len(self.index_batches)))

        for index in range(len(self.index_batches)):
            #print(self.index_batches, "AAAAAAAAAAA")
            #print(len(self.index_batches))
            try:
                if index + self.set_size + self.label_size >= len(self.index_batches):
                    break
                sample = self.index_batches[index: index + self.set_size + self.label_size]
                #print(sample)
                valid = True
                length = len(sample)
                tsi = self.headers.index("TS")
                if index % 100 == 0:
                    self.echo.buttom_print("[dataset] generating index data at {}, {} sample generated".format(
                        time.strftime(
                        "%y-%m-%d %H:%M:%S", time.localtime(sample[0][self.headers.index("TS")] / 1000)),
                        len(self.index_valids)))
                for i, e in enumerate(sample):
                    if i + 1 >= length:
                        break
                    if e[tsi] + self.sample_time != sample[i + 1][tsi]:
                        #print(e)
                        #print(sample)
                        #self.echo.print("[database] divided sample detected at {} and {} for {}ms".format(e[tsi],
                        #                                                                   sample[i + 1][tsi],
                        #                                                                sample[i + 1][tsi]-e[tsi]))
                        valid = False
                        break
                if not valid:
                    continue
                val = {"TS": sample[0][tsi], "OFFSET": index}
                #print(val)
                self.index_valids.append(val)
            except KeyboardInterrupt:
                self.echo.print("[dataset] process interrupted by keyboard")

        #del self.index_batches
        #self.index_batches = None
        self.echo.print("[dataset] \n{}".format(self.index_valids.head(3)))
        self.echo.buttom_print("")
        self.echo.print("[dataset] generated {} continues samples".format(len(self)))

        #if self.index_file is not None:
        #    self.dump_index()

    def dump_index(self, filename=None):
        if filename is None:
            filename = self.index_file
        headers_index = np.array(self.headers)
        np.savez_compressed(filename, headers_index=headers_index,
                            batches=self.index_batches.data, valids=self.index_valids.data)

    def read_index(self, filename=None):
        if filename is None:
            filename = self.index_file
        raw = np.load(filename, allow_pickle=True)
        self.index_batches = Dataframe(data=raw["batches"], header=raw["headers_index"].tolist())
        self.index_valids = Dataframe(data=raw["valids"], header=["TS", "OFFSET"])
        self.indexed = True

    def get_indexs(self):
        return list(range(len(self.index_valids)))

    def get_batch(self, index):
        #print(index)
        index = self.index_batches[index]
        ret = {}
        for ele in self.db_api.monitoring:
            data = {"input_1": [],
                    "input_A": [],
                    "input_B": []}
            raw = [list(self.db_api.all_tables[ele].get(i)[0]) for i in index[self.headers.index(ele)]]
            kopen = raw[0][4]
            kclose = raw[-1][4]
            kline = [e[4] for e in raw]
            khigh = max(kline)
            klow = min(kline)
            buy_count = sum([e[9] for e in raw])
            buy_amount = sum([e[10] for e in raw])
            sell_count = sum([e[13] for e in raw])
            sell_amount = sum([e[14] for e in raw])
            buy_max = max([e[11] for e in raw])
            #buy_min = min([e[12] for e in raw])
            sell_max = max([e[15] for e in raw])
            #sell_min = min([e[16] for e in raw])
            amount = buy_amount + sell_amount
            depth_0_buy = json.loads(raw[-1][17])
            depth_0_sell = json.loads(raw[-1][18])
            depth_5_buy = json.loads(raw[-1][19])
            depth_5_sell = json.loads(raw[-1][20])
            b0 = [0.0 for e in range(150)]
            b5 = [0.0 for e in range(20)]
            s0 = [0.0 for e in range(150)]
            s5 = [0.0 for e in range(20)]
            for i, e in enumerate(depth_0_buy):
                b0[i] = e[1]
            for i, e in enumerate(depth_0_sell):
                s0[i] = e[1]
            for i, e in enumerate(depth_5_buy):
                b5[i] = e[1]
            for i, e in enumerate(depth_5_sell):
                s5[i] = e[1]
            data["input_1"] = [kopen, kclose, klow, khigh, buy_count, buy_amount, sell_count, sell_amount,
                               buy_max, sell_max, amount]
            data["input_A"] = b0[::-1] + s0
            data["input_B"] = b5[::-1] + s5
            ret.update({ele: data})

        return ret

    def get_feature(self, index, key=None):  # data pre-processing function
        tradename = TARGETMARKET
        start_offset = self.index_valids[index, "OFFSET"]
        input_1 = Dataframe(dtype="float32")
        input_A = Dataframe(dtype="float32")
        input_B = Dataframe(dtype="float32")
        #print(start_offset)
        for i in range(self.set_size):
            raw = self.get_batch(start_offset + i)
            in1 = []
            for ele in self.db_api.monitoring:
                in1 += raw[ele]["input_1"]
            input_1.append(in1)
            input_A.append(raw[tradename]["input_A"])
            input_B.append(raw[tradename]["input_B"])
        std = input_1.data.std(axis=0)
        std[std == 0] = 1
        input_1.data = (input_1.data - input_1.data.mean(axis=0)) / std
        std = input_B.data.std(axis=0)
        std[std == 0] = 1
        input_B.data = (input_B.data - input_B.data.mean(axis=0)) / std
        std = input_A.data.std(axis=0)
        std[std == 0] = 1
        input_A.data = (input_A.data - input_A.data.mean(axis=0)) / std
        ret = {"input_1": input_1.data,
               "input_B": input_B.data,
               "input_A": input_A.data}
        if key is not None:
            ret = ret[key]
        return ret

    def get_label(self, index):  # data pre-processing function
        tradename = TARGETMARKET
        start_offset = self.index_valids[index, "OFFSET"] + self.set_size
        kline = []
        zero = self.get_batch(start_offset - 1)[tradename]["input_1"][1]
        for i in range(self.label_size):
            raw = self.get_batch(start_offset + i)
            kline.append(raw[tradename]["input_1"][1])
        ret = np.array(kline, dtype="float32")
        ret = ret - zero
        return ret


def get_input_1(index):
    global DATASET
    if not isinstance(DATASET, DatasetBase):
        raise Exception("invalid Dataset Base")
    return DATASET.get_feature(index, "input_1")


def get_input_A(index):
    global DATASET
    if not isinstance(DATASET, DatasetBase):
        raise Exception("invalid Dataset Base")
    return DATASET.get_feature(index, "input_A")


def get_input_B(index):
    global DATASET
    if not isinstance(DATASET, DatasetBase):
        raise Exception("invalid Dataset Base")
    return DATASET.get_feature(index, "input_B")


def rnn_init():
    global ETH_RNN, DATASET

    paths = [MODEL_FILE]
    for i in paths:
        path_fixer(i)

    print("initializing database...")

    db_api = MarketDB()
    DATASET = DatasetBase(db_api, sample_time_ms=SAMPLE_TIME_MS, set_size=SAMPLE_SIZE, label_size=PREDICT_SIZE,
                          use_index_json=True, index_json=DATASET_INDEX)
    DATASET.init_dataset()


def main():
    global DATASET
    # 初始化神经网络
    if not isinstance(DATASET, DatasetBase):
        raise Exception("dataset base is not ready yet")

    seed = 10
    rnn = customNN(NAME)

    index_list = DATASET.get_indexs()
    random.seed(seed)
    random.shuffle(index_list)

    test_list = index_list[:int(len(index_list) * TEST_RATE)]
    train_list = index_list[int(len(index_list) * TEST_RATE):]

    print("generated {} train samples and {} test samples".format(len(train_list), len(test_list)))

    dset_input_1 = tf.data.Dataset.from_tensor_slices({"input_1": train_list})
    dset_input_1 = dset_input_1.map(get_input_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_input_A = tf.data.Dataset.from_tensor_slices({"input_A": train_list})
    dset_input_A = dset_input_A.map(get_input_A, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_input_B = tf.data.Dataset.from_tensor_slices({"input_B": train_list})
    dset_input_B = dset_input_B.map(get_input_B, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_label = tf.data.Dataset.from_tensor_slices(train_list)
    dset_label = dset_label.map(DATASET.get_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((dset_input_1, dset_input_A, dset_input_B))
    train_dataset = tf.data.Dataset.zip((train_dataset, dset_label))
    train_dataset = train_dataset.shuffle(buffer_size=int(2048 * BUFF_RATE)).batch(BATCH_SIZE)

    dset_input_1_test = tf.data.Dataset.from_tensor_slices({"input_1": test_list})
    dset_input_1_test = dset_input_1_test.map(get_input_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_input_A_test = tf.data.Dataset.from_tensor_slices({"input_A": test_list})
    dset_input_A_test = dset_input_A_test.map(get_input_A, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_input_B_test = tf.data.Dataset.from_tensor_slices({"input_B": test_list})
    dset_input_B_test = dset_input_B_test.map(get_input_B, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_label_test = tf.data.Dataset.from_tensor_slices(test_list)
    dset_label_test = dset_label_test.map(DATASET.get_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.zip((dset_input_1_test, dset_input_A_test, dset_input_B_test))
    test_dataset = tf.data.Dataset.zip((test_dataset, dset_label_test))
    test_dataset = test_dataset.shuffle(buffer_size=int(2048 * BUFF_RATE)).batch(BATCH_SIZE)

    rnn.train_db = train_dataset
    rnn.test_db = test_dataset

    # 初始化网络模型并执行设置
    if os.path.exists(MODEL_FILE):
        rnn.load_model(MODEL_FILE)
        print("loaded model file from \"{}\"".format(MODEL_FILE))
    else:
        rnn.init_model()

    #print(rnn.model.summary())


    rnn.enable_tensorboard()
    # cnn.enable_checkpointAutosave(MODEL_FILE)

    # 初次训练网络
    #choice = input("start training for {} epoch(s)? (Y/n): ".format(str(EPOCHES)))
    choice = "Y"
    trained = True
    if EPOCHES > 0 and choice in ("Y", "y", "yes"):
        rnn.train(epochs=EPOCHES)
        trained = True

    # 保存模型
    if trained:
        rnn.save_model(MODEL_FILE)
        print("model saved to \"{}\"".format(MODEL_FILE))

    # 测试模型
#    print("evaluating trained model...")
#    rnn.evaluate()


if __name__ == "__main__":
    rnn_init()
    main()
else:
    #rnn_init()
    pass
