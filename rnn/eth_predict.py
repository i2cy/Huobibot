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
import pandas as pd
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

DATASET_INDEX = "database/index.json"  # 数据集预处理索引ID

TEST_RATE = 0.05  # 测试集比例
BATCH_SIZE = 10  # 批处理大小
SAMPLE_SIZE = 1800  # 特征样本大小
PREDICT_SIZE = 300  # 预测输出大小
SAMPLE_TIME_MS = 3000  # 采样间隔时间（单位ms）
EPOCHES = 10  # 训练代数
BUFF_RATE = 0.1  # 缓冲区大小指数
LEARNING_RATE = 0.0001  # 学习率
MODEL_FILE = "models/eth_market_model.h5"  # 在此处修改神经网络模型文件
NAME = "CryptoCoinPrediction"

ETH_RNN = None


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

        input_1 = tf.keras.Input(shape=(SAMPLE_SIZE, 13), name="input_1")
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
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                           loss="mse",  # 均方差预测问题
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
                                                steps_per_epoch=self.train_size // self.batch_size,
                                                validation_data=self.test_db,
                                                validation_steps=self.test_size // self.batch_size,
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


def read_preprocess_image(img_path):  # 定义数据集map函数
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 300, 300)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img = img / 255  # 图像归一化，使得输入数据在（-1,1）区间范围内
    return img


def read_preprocess_image_from_raw(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 300, 300)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img = img / 255  # 图像归一化，使得输入数据在（-1,1）区间范围内
    return img


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


class DatasetBase:

    def __init__(self, market_db_api, sample_time_ms=3000, set_size=20 * 15,
                 echo=None, index_json=DATASET_INDEX, use_index_json=True):
        if not isinstance(market_db_api, MarketDB):
            raise TypeError("market database API must be a MarketDB object")
        if not isinstance(echo, Echo):
            echo = self.__internal_echo__
        self.indexed = use_index_json
        if not os.path.exists(index_json):
            self.indexed = False
        self.index_file = index_json
        self.echo = echo
        self.db_api = market_db_api
        self.sample_time = sample_time_ms
        self.set_size = set_size
        self.index_batches = pd.DataFrame(None, columns=["TS"] + self.db_api.monitoring)
        self.index_features = pd.DataFrame(None, columns=["TS"] + self.db_api.monitoring)
        self.index_labels = pd.DataFrame(None, columns=["TS"] + self.db_api.monitoring)
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
        return len(self.index_features)

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
                self.index_batches = self.index_batches.append(data, ignore_index=True)
            offset += self.sample_time

        self.echo.print("[dataset] \n{}".format(self.index_batches.head(5)))
        self.echo.buttom_print("")

        self.echo.print("[dataset] proceed {} samples".format(len(self.index_batches)))

        for index, ele in enumerate(self.index_batches.iloc):
            try:
                if index + 2 * self.set_size >= len(self.index_batches):
                    break
                sample = self.index_batches.iloc[index: index + 2 * self.set_size]
                valid = True
                length = len(sample)
                if index % 100 == 0:
                    self.echo.buttom_print("[dataset] generating index data at {}".format(time.strftime(
                        "%y-%m-%d %H:%M:%S", time.localtime(sample.iloc[0]["TS"] / 1000))))
                for i, e in enumerate(sample.iloc):
                    if i + 1 >= length:
                        break
                    if e["TS"] + self.sample_time != sample.iloc[i + 1]["TS"]:
                        valid = False
                        break
                if not valid:
                    continue
                feature_set = self.index_batches.iloc[index: index + self.set_size]
                label_set = self.index_batches.iloc[index + self.set_size: index + 2 * self.set_size]
                self.index_features = self.index_features.append(feature_set, ignore_index=True)
                self.index_labels = self.index_labels.append(label_set, ignore_index=True)
            except KeyboardInterrupt:
                self.echo.print("[dataset] process interrupted by keyboard")

        del self.index_batches
        self.index_batches = None
        self.echo.print("[dataset] generated {} continues samples".format(len(self)))

        #if self.index_file is not None:
        #    self.dump_index()

    def dump_index(self, filename=None):
        if filename is None:
            filename = self.index_file
        raw = {"features": self.index_features.to_dict(),
               "labels": self.index_labels.to_dict()}
        with open(filename, "w") as f:
            json.dump(raw, f)

    def read_index(self, filename=None):
        if filename is None:
            filename = self.index_file
        with open(filename, "r") as f:
            raw = json.load(f)
        self.index_features = pd.DataFrame(raw["features"])
        self.index_labels = pd.DataFrame(raw["labels"])
        self.indexed = True

    def get_indexs(self):
        return list(self.index_features.index)

    def get_feature(self, index, tradename=None):  # data pre-processing function
        if tradename is None:
            ret = {}
            for ele in self.db_api.monitoring:
                data = {"input_1": [],
                        "input_A": [],
                        "input_B": []}
                index_list = self.index_features.iloc[index][ele]
                raw = [list(self.db_api.all_tables[ele].get(i)[0]) for i in index_list]
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
                buy_min = min([e[12] for e in raw])
                sell_max = max([e[15] for e in raw])
                sell_min = min([e[16] for e in raw])
                amount = sum([e[9]+e[13] for e in raw])
                depth_0_buy = json.loads(raw[-1][17])
                depth_0_sell = json.loads(raw[-1][18])
                depth_5_buy = json.loads(raw[-1][19])
                depth_5_sell = json.loads(raw[-1][20])
                b0 = [0.0 for e in range(150)]
                b5 = [0.0 for e in range(150)]
                s0 = [0.0 for e in range(150)]
                s5 = [0.0 for e in range(150)]
                for i, e in enumerate(depth_0_buy):
                    b0[i] = e[1]
                for i, e in enumerate(depth_0_sell):
                    s0[i] = e[1]
                for i, e in enumerate(depth_5_buy):
                    b5[i] = e[1]
                for i, e in enumerate(depth_5_sell):
                    s5[i] = e[1]
                data["input_1"] = [kopen, kclose, klow, khigh, buy_count, buy_amount, sell_count, sell_amount,
                                   buy_max, buy_min, sell_max, sell_min, amount]
                data["input_A"] = b0[::-1] + s0
                data["input_B"] = b5[::-1] + s5
                ret.update({ele: data})



    def get_label(self, index):  # data pre-processing function
        pass


def rnn_init():
    global ETH_RNN

    paths = [MODEL_FILE]
    for i in paths:
        path_fixer(i)

    print("initializing...")

    if __name__ != "__main__":
        CAPTCHA_CNN = customNN(NAME)
        CAPTCHA_CNN.load_model(MODEL_FILE)
        print("captcha CNN fully connected")


def main():
    img_paths = [str(ele) for ele in data_root.glob("*.jpg")]

    labels = []
    label_to_word = {}
    for i in img_paths:
        i = i.split('/')[-1]
        name = i.split("_")
        id = int(name[1])
        label_to_word.update({id: name[0]})
        labels.append(id)

    print("dumping label_to_word...")
    json.dump(label_to_word, open(DICT_FILE, 'w'), indent=2)

    img_counts = len(img_paths)
    print("loaded", img_counts, "image paths")

    label_counts = len(labels)
    print("loaded", label_counts, "labels")

    print("dataset head:")
    print("============================================" * 3)
    print("Label\t\tIMG_Path")
    for i, l in enumerate(labels):
        if i > 9:
            break
        print(" " + str(labels[i]) + "\t\t" + str(img_paths[i]))
    print("============================================" * 3)

    # 初始化神经网络
    cnn = customNN(NAME)
    cnn.load_dataset((img_paths, labels),
                     mapFunc=read_preprocess_image,
                     testRate=TEST_RATE,
                     batchSize=BATCH_SIZE,
                     shufflePercentage=BUFF_RATE
                     )

    # 初始化网络模型并执行设置
    if os.path.exists(MODEL_FILE):
        cnn.load_model(MODEL_FILE)
        print("loaded model file from \"{}\"".format(MODEL_FILE))
    else:
        cnn.init_model()

    print(cnn.model.summary())

    cnn.set_model_file(MODEL_FILE)

    print("testing model speed...")

    val_root = pathlib.Path("../validations")

    test_files = [str(ele) for ele in val_root.glob("*.jpg")]

    test_data = [read_preprocess_image(ele) for ele in test_files]

    for i in range(5):
        t1 = time.time()
        for ele in test_data:
            res = cnn.predict(ele)
        t2 = time.time()
        print("  testing {} files, time spent {}ms, {}ms per pred".format(len(test_data),
                                                                          round((t2 - t1) * 1000, 2),
                                                                          round(((t2 - t1) / len(test_data)) * 1000,
                                                                                2)))

    print("outputs: {}".format(res))

    for i in range(5):
        t3 = time.time()
        data = tf.data.Dataset.from_tensor_slices(test_data).repeat().batch(8)
        t4 = time.time()
        t1 = time.time()
        res = cnn.predict(data.take(1))
        t2 = time.time()
        print("  testing {} files, time spent {}ms, {}ms per data,\n  batching spent {}ms, total {}ms".format(
            len(test_data),
            round((t2 - t1) * 1000, 2),
            round((t2 - t1) * 125, 2),
            round((t4 - t3) * 1000, 2),
            round((t2 - t3) * 1000, 2)))

    print("outputs: {}".format(res))

    cnn.enable_tensorboard()
    # cnn.enable_checkpointAutosave(MODEL_FILE)

    # 检查数据集匹配是否有错
    print("datasets:\n{}".format(str(cnn.train_db)))
    for i in range(3):
        img = read_preprocess_image(img_paths[random.randint(0, img_counts)])
        plt.imshow(img)
        plt.show()

    # 微调模型
    if True:
        choice = input("should we fine tune now? level(1,2/n): ".format(str(EPOCHES)))
        if EPOCHES > 0 and choice in ("1", "2"):
            cnn.postProc_model(int(choice))

    # 初次训练网络
    choice = input("start training for {} epoch(s)? (Y/n): ".format(str(EPOCHES)))
    trained = False
    if EPOCHES > 0 and choice in ("Y", "y", "yes"):
        cnn.train(epochs=EPOCHES)
        trained = True

    # 保存模型
    if trained:
        cnn.save_model()
        print("model saved to \"{}\"".format(MODEL_FILE))

    # 测试模型
    print("evaluating trained model...")
    cnn.evaluate()

    # 预测
    while True:
        path = input("test file path: ")
        res = cnn.predict(read_preprocess_image(path))
        word = label_to_word[tf.argmax(res[0]).numpy()]
        print("result: {}".format(word))
        plt.imshow(read_preprocess_image(path))
        plt.show()


if __name__ == "__main__":
    rnn_init()
    main()
else:
    # rnn_init()
    pass
