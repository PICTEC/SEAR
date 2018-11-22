import csv
import hickle
import logging
import multiprocess
import numpy as np
import os
import pickle
import random
import scipy.io.wavfile as sio
import sys
import types

import time

from loaders.utils import ProgressBar

loader_pool = None
manager = None
global_dict = {}

def enable_multiprocessing():
    import pathos
    global loader_pool, manager, global_dict
    manager = multiprocess.Manager() # must be before
    global_dict = manager.dict()
    loader_pool = pathos.multiprocessing.Pool(8)

def disable_multiprocessing():
    global loader_pool, manager, global_dict
    manager = None
    global_dict = {}
    loader_pool.terminate()
    loader_pool = None


class Cache(object):
    # TODO: move to utils

    ram_cache = None
    ram_cache_size = None
    ram_current_usage = None
    _ram_units = {'k': 1024, "K": 1024, "M": 1024 ** 2, "m":1024 ** 2, "G":1024 ** 3, "g":1024 ** 3}
    _cache_ready = False
    dict = {}

    def __init__(self, name, function, cache, uses_ram=False):
        self.name = name
        self.function = function
        self.cache = cache
        if not os.path.exists(self.cache):
            os.mkdir(self.cache)
        self.uses_ram=uses_ram
        logging.log(11, "Cache created")

    @classmethod
    def set_memory_pool(cls, value):
        cls.cache_ram_usage = 0
        if type(value) == str:
            if value[-1] in cls._ram_units.keys():
                value = int(float(value[:-1]) * cls._ram_units[value[-1]])
        cls.cache_ram_limit = value
        cls._cache_ready = True
        logging.log(11, "RAM set to usable, limit: {}".format(value))

    def get_or_none(self, cache_key):
        try:
            if self.uses_ram and Cache._cache_ready:
                t = time.time()
                data = Cache.ram_get(cache_key)
                logging.log(11, "Get {}: {}".format(self.name, time.time() - t))
                return data
            else:
                path = os.path.join(self.cache, cache_key)
                if os.path.exists(path + ".npb"):
                    return np.fromfile(path + ".npb", dtype=np.float32)
                else:
                    with open(path + ".pkl", 'rb') as f:
                        return pickle.load(f)
        except (IOError, KeyError):
            return None

    def add(self, cache_key, data):
        if self.uses_ram and Cache._cache_ready:
            Cache.ram_put(cache_key, data)
        else:
            path = os.path.join(self.cache, cache_key)
            if isinstance(data, np.ndarray):
                data.tofile(path + ".nbp")
            else:
                with open(path + ".pkl", 'wb') as f:
                    pickle.dump(data, f)

    @classmethod
    def ram_put(cls, cache_key, data):
        # assumes is numpy array
        size = cls.cache_ram_usage + sys.getsizeof(data)
        logging.log(11, "{}, key entering: {}".format(size, cache_key))
        if cls.cache_ram_limit > size and cache_key not in cls.dict.keys():
            cls.cache_ram_usage = size
            cls.dict[cache_key] = data

    @classmethod
    def ram_get(cls, cache_key):
        logging.log(11, "Getting key: {}".format(cache_key))
        return cls.dict[cache_key]


class Dataset(object):
    """
    The class is responsible for creating a list of sequences padded for training
    """

    enable_dirty_GSM_hack = False

    def __init__(self, files, root=".", cache=None, metadata=None, verbose=False, dataset_pad = None, trim_lengths = None, ram_cache_size = None,
        enable_dirty_GSM_hack=False):
        self.files = files
        self.root = root  # where data lies
        self.cache = cache if cache else root # where examples are stored
        self.inputs = []
        self.outputs = []
        self.extra_sources = {}
        self.metadata = metadata
        self.verbose = verbose
        self.dataset_pad = dataset_pad # for dimensionality reduction in time
        self.trim_lengths = trim_lengths
        self.input_normalizers = []
        self.output_normalizers = []
        self.pb = ProgressBar() if verbose else None
        self.ram_cache_size = ram_cache_size
        self._full_cache = None
        Dataset.enable_dirty_GSM_hack = enable_dirty_GSM_hack

    @staticmethod
    def from_folder(path, filter=None, **kwargs):
        files = os.listdir(path)
        if filter is not None:
            files = [x for x in files if filter(x)]
        return Dataset(files, root=path, **kwargs)

    @staticmethod
    def from_file(path, ignore_first_line = False, **kwargs):
        # TODO: common root
        with open(path) as f:
            files = list(csv.reader(f))
            if ignore_first_line:
                files = files[1:]
            records = zip(*files)
        files = records[0]
        metadata = dict(files, records[1:])
        return Dataset(files, metadata, **kwargs)

    def from_files(self, files):
        """
        This method actually replaces the files of the dataset and allows to produce
        new data using the same transforms
        """
        self.files = files
        self.root = os.getcwd()
        return self

    def add_input(self, transform, normalizer = None):
        self.input_normalizers.append(normalizer)
        self.extra_sources.update({k:v(self) for k,v in transform.make_sources().items()})
        self.inputs.append((transform.source, transform)) # transform.__call__

    def add_output(self, transform, normalizer = None):
        self.output_normalizers.append(normalizer)
        self.extra_sources.update({k:v(self) for k,v in transform.make_sources().items()})
        self.outputs.append((transform.source, transform)) # transform.__call__

    def _sources(self, get):
        if get == "signal":
            def signal(f):
                data = sio.read(os.path.join(self.root, f))[1]
                dtype = data.dtype
                if self.trim_lengths is not None:
                    data = data[:self.trim_lengths]
                if dtype != np.float32:
                    data = data.astype(np.float32)
                    data /= (2 ** 15) if dtype == np.int16 else (2 ** 8 - 1)
                return data
            return signal
        elif get == "fname":
            return lambda f:f
        elif get == "fpath":
            return lambda f:os.path.join(self.root, f)
        elif get.startswith("meta:"):
            index = int(get.split(":")[1])
            return lambda f: self.metadata[f][index]
        elif get.startswith("extra:"):
            return self.extra_sources[get]
        else:
            raise ValueError("Cannot get source: {}".format(get))

    def _extract_file_maker(self):
        input_types = self.inputs
        output_types = self.outputs
        verbose = self.verbose
        caches = {source[0]: self._sources(source[0]) for source in (input_types + output_types)}
        def extract(file):
            inputs, outputs = [], []
            if verbose:
                self.pb.step(file)
            for source, transform in input_types:
                inputs.append(transform(caches[source](file)))
            for source, transform in output_types:
                outputs.append(transform(caches[source](file)))
            return inputs, outputs
        return extract

    def _extract_filelist(self, filelist):
        """
        Get all examples which are in cache
        For each example not in cache, run extract
        Add missing examples to cache
        """
        filelist = list(filelist)
        if self._full_cache is None:
            self._full_cache = Cache("full", ..., self.cache, bool(self.ram_cache_size))
        data = {x: self._full_cache.get_or_none(x) for x in filelist}
        missing_dict = {x: data[x] for x in filelist if data[x] is None}
        missing = [name for name in filelist if data[name] is None]
        if loader_pool is not None:
            missing = self._normalize(loader_pool.imap(self._extract_file_maker(), missing)) # TODO: use self.pool after cache is in shared memory
        else:
            missing = self._normalize(map(self._extract_file_maker(), missing)) # TODO: use self.pool after cache is in shared memory
        for k, v in zip(missing_dict.keys(), missing):
            self._full_cache.add(k, v)
            missing_dict[k] = v
        data.update(missing_dict)
        return [data[x] for x in filelist]

    def produce(self, valid_ratio = 0.1, test_ratio = 0.15, train_data = None, valid_data = None, test_data = None):
        # TODO: by numbers, not percentages
        files = self.files[:]
        random.shuffle(files)
        if all([x is None for x in [train_data, valid_data, test_data]]):
            test_begin = int(len(files) * (1 - test_ratio))
            valid_begin = int(len(files) * (1 - test_ratio - valid_ratio))
            train_files = files[:valid_begin]
            valid_files = files[valid_begin:test_begin]
            test_files = files[test_begin:]
        else:
            train_files = files[:train_data]
            valid_files = files[train_data:(train_data + valid_data)]
            test_files = files[(train_data + valid_data):(train_data + valid_data + test_data)]
        Cache.set_memory_pool(self.ram_cache_size)
        self._train_normalizers(train_files)
        if self.pb is not None: self.pb.set_max(len(train_files))
        train_data = Dataset.pad_n_stack(self._extract_filelist(train_files), divisible_pad = self.dataset_pad)
        if self.pb is not None: self.pb.set_max(len(valid_files))
        valid_data = Dataset.pad_n_stack(self._extract_filelist(valid_files), divisible_pad = self.dataset_pad)
        if self.pb is not None: self.pb.set_max(len(test_files))
        test_data = Dataset.pad_n_stack(self._extract_filelist(test_files), divisible_pad = self.dataset_pad)
        return train_data, valid_data, test_data

    def produce_single(self):
        files = self.files[:]
        return Dataset.pad_n_stack(self._extract_filelist(files), divisible_pad = self.dataset_pad)

    @staticmethod
    def _pad(X, divisible_pad=None):
        #print([x.shape for x in X])
        if all([isinstance(x, list) for x in X]):
            X = [np.array(x) for x in X]
        if all([isinstance(x, int) for x in X]):
            return np.array(X, np.int) # need to check types
        maxlen = max([x.shape[0] for x in X])
        if divisible_pad is not None:
            offset = divisible_pad - maxlen % divisible_pad
            if offset == divisible_pad:
                offset = 0
        else:
            offset = 0
        if len(X[0].shape) == 2:
            return [np.pad(item, ((0, offset + maxlen - item.shape[0]), (0, 0)), 'constant') for item in X]
        elif len(X[0].shape) == 1:
            return [np.pad(item, ((0, offset + maxlen - item.shape[0]),), 'constant') for item in X]

    @classmethod
    def pad_n_stack(cls, generator, divisible_pad = None):
        """
        X = list of examples, each a list of nparrays, one per transform
        y = list of examples, each a list of nparrays, one per transform
        Target:
        X <- np.array or list of np.arrays
        y <- np.array or list of np.arrays
        """
        X, y = zip(*(i for i in generator))
        # logging.log(11, (type(X), type(X[0]), type(X[0][0]), len(X), len(X[0]), len(X[0][0]))) # debug
        if len(X[0]) == 1:
            X = np.stack(Dataset._pad([i[0] for i in X], divisible_pad))
        else:
            X = [np.stack(Dataset._pad([x[inp] for x in X], divisible_pad)) for inp in range(len(X[0]))]
        if len(y[0]) == 1:
            y = np.stack(Dataset._pad([i[0] for i in y]))
        else:
            y = [np.stack(Dataset._pad([x[inp] for x in y])) for inp in range(len(y[0]))]
        if cls.enable_dirty_GSM_hack:
            X = X[:, :min([X.shape[1], y.shape[1]]), :]
            y = y[:, :min([X.shape[1], y.shape[1]]), :]
        return X, y

    def iterator(self, batch_size, valid_ratio = 0.1, test_ratio = 0.15, train_data = None, valid_data = None, test_data = None):
        files = self.files[:]
        random.shuffle(files)
        if all([x is None for x in [train_data, valid_data, test_data]]):
            test_begin = int(len(files) * (1 - test_ratio))
            valid_begin = int(len(files) * (1 - test_ratio - valid_ratio))
            train_files = files[:valid_begin]
            valid_files = files[valid_begin:test_begin]
            test_files = files[test_begin:]
        else:
            train_files = files[:train_data]
            valid_files = files[train_data:(train_data + valid_data)]
            test_files = files[(train_data + valid_data):(train_data + valid_data + test_data)]
        Cache.set_memory_pool(self.ram_cache_size)
        self._train_normalizers(train_files)
        return [self._make_iterator(x, batch_size) for x in [train_files, valid_files, test_files]]

    def _make_iterator(self, file_list, batch_size):


        class Generator(object):

            _extract_filelist = lambda _, x: self._extract_filelist(x)

            def __init__(self, file_list, dataset_pad):
                self.file_list = file_list
                self.dataset_pad = dataset_pad
                random.shuffle(self.file_list)
                self.file_iter = iter(self.file_list)

            def __len__(self):
                return len(self.file_list) // batch_size

            def __iter__(self):
                random.shuffle(self.file_list)
                self.file_iter = iter(self.file_list)
                return self
            
            def __next__(self):
                try:
                    list = [next(self.file_iter) for x in range(batch_size)]
                    list = self._extract_filelist(list)
                    data = Dataset.pad_n_stack(list, divisible_pad = self.dataset_pad) # if slow, this stmt is to blame
                    return data
                except StopIteration:
                    random.shuffle(self.file_list)
                    self.file_iter = iter(self.file_list)
                    return next(self)

        return Generator(file_list, self.dataset_pad)

    def _train_normalizers(self, train_files):
        # Bootstrap identity self._normalize()
        # gather all trainable normalizers
        # fit all of them
        # bootstrap real _normalize()
        # TODO: support for already trainable normalizers
        def null_normalizer(self, x):
            return x
        self._normalize = types.MethodType(null_normalizer, self)
        normalizers = []
        normalizers += [lambda data: x(data[0][ix]) for ix, x in enumerate(self.input_normalizers) if x is not None] 
        normalizers += [lambda data: x(data[1][ix]) for ix, x in enumerate(self.output_normalizers) if x is not None]
        logging.info("Training normalizers...")
        logging.log(11, normalizers)
        if normalizers:
            for chunk in self._extract_filelist(train_files):
                for normalizer in normalizers:
                    normalizer.partial_fit(chunk)
            def _normalize(self, input_data):
                normalized = []
                for item in input_data:
                    transformed = []
                    for subset, normset in zip(item, [self.input_normalizers, self.output_normalizers]):
                        fooput = []
                        for normalizer, feature in zip(normset, subset):
                            if normalizer is not None:
                                shape = feature.shape
                                feature = normalizer(feature.reshape(-1, shape[-1]))
                                fooput.append(feature.reshape(*shape))
                            else:
                                fooput.append(feature)
                        transformed.append(fooput)
                    normalized.append(transformed)
                return normalized
            self._normalize = types.MethodType(_normalize, self)

    def make_prototype(self):
        return self  # should it be like that?

    def iterate_over_files(self, batch_size):
        n_files = len(self.files)
        n_batches = n_files // batch_size + (1 if n_files % batch_size else 0)
        def generator():
            for i in range(n_batches):
                list = self.files[i * batch_size : (i + 1) * batch_size]
            list = self._extract_filelist(list)
            yield Dataset.pad_n_stack(list, divisible_pad=self.dataset_pad)
        return generator()

    def all_names(self):
        return [os.path.join(self.root, x) for x in self.files]



def from_hickle(fname, test_name):  # from what shall we use testing and validation?
    train = hickle.load(fname)
    valid = None
    test = hickle.load(test_name)
    return train, valid, test
