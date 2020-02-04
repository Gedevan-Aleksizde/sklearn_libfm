import os
import tempfile
import shutil
import numpy as np
# import scipy as sp
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file
import pandas as pd

class LibFmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, validation=None,
                 method='als', iter=100, cache_size='infty', k0=1, k1=1, k2=8, init_stdev=0.1,
                 learn_rate=0.1, regular_k0=0, regular_k1=0, regular_k2=0, meta=None,
                 relation=None, random_state=None,
                 verbosity=0, verbose=None, quiet=True, libpath=None, rlog=False, **kwargs):
        """
        This is an unofficial lifFM wrapper for scikit-learn ecosystems.
        Not used: train, test, out.
        More information for official libFM manual: http://www.libfm.org/
        :param method:
        :param iter:
        :param cache_size:
        :param k0: integer or boolean. bias term, the 1st element of `-dim` in the original libfm's argument. ; default: 1
        :param k1: integer or boolean. linear term. original is the second element of `-dim`; default: 1
        :param k2: integer or boolean. dimension of pairwise interaction term, the third element of `-dim`; default: 8
        :param init_stdev: for initialization of 2-way factor; default: 0.1
        :param learn_rate: learning rate for SGD; default: 0.1
        :param regular_k0: default: 0
        :param regular_k1: default: 0
        :param regular_k2: default: 0
        :param meta:
        :param relation:
        :param random_state: equivalent to `seed` default : None
        :param validation:
        :param verbosity: 0 or more.
        :param verbose: alias of `verbosity`. `verbose` is prior if you specify both of them.
        :param quiet: boolean. surpress std output; default: True
        :param libpath: if you don't add `libfm/bin` to PATH, you must specify this.
        :param rlog: NOTE: this class use dummy test data. its performance is meaningless; default; False
        :param kwargs: dict. other unknown parameters that the official manual doesn't say. NOTE: It's almost enough to do with the above keywords.
        """

        self.cache_size = cache_size
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.init_stdev = init_stdev
        if method not in ['sgd', 'sgda', 'als']:
            raise('Argument `method` requires either of the followings: `sgd`, `sgda`, `als`. ALL SMALL CASE REQUIRED :(')
        else:
            self.method = method
        self.iter = iter
        self.learn_rate = learn_rate
        self.meta = meta
        self.regular_k0 = regular_k0
        self.regular_k1 = regular_k1
        self.regular_k2 = regular_k2
        self.random_state = random_state
        # task not used: specify LibFmClassifier or LibFmRegressor
        self.rlog = rlog
        self.relation = relation
        self.random_state=random_state
        self.validation = validation
        if method == 'SGD' and self.validation is None:
            raise('Argument `validation` needed if you specify "SGD"')
        if verbosity and verbose:
            print('Argument `verbose` is an alias of `verbosity`. `verbose` accepted.')
            self.verbosity = verbose
        elif verbosity or verbose:
            self.verbosity = verbosity if verbose is not None else verbosity
        else:
            self.verbosity = 0
        self.libpath = None if libpath is None else libpath.__str__()
        if shutil.which('libFM', path=self.libpath) is None:
            Warning('libFM command not found')
        if self.verbosity > 0:
            print('Argument `verbose` more than zero but `quiet` is True. turned it to False.')
            self.quiet = False
        else:
            self.quiet = quiet
        self.kwargs = kwargs

    def fit(self, X, y):
        params = {
            'dim': "'" + ','.join([str(int(x)) for x in [self.k0, self.k1, self.k2]]) + "'",
            'method': self.method,
            'iter': self.iter,
            'cache_size': self.cache_size,
            'init_stdev': self.init_stdev,
            'learn_rate': self.learn_rate,
            'seed': self.random_state,
            'meta': self.meta,
            'regular': "'" + ','.join([str(x) for x in [self.regular_k0, self.regular_k1, self.regular_k2]]) + "'",
            'relation': self.relation,
            'verbosity': None if self.verbosity == -1 else self.verbosity
        }
        params.update(self.kwargs)
        params = {k: str(v) for k, v in params.items() if v is not None}
        # TODO: for Windows
        # TODO: libsvm の formatでのラベルの扱い
        self.classes_ = np.unique(y)
        with tempfile.TemporaryDirectory() as tempdir:
            params['train'] = Path(tempdir).joinpath('train_data.dat').__str__()
            with Path(params['train']).open('wb') as train_path:
                dump_svmlight_file(X, y, train_path)
            params['test'] = Path(tempdir).joinpath('test_data.dat').__str__()
            with Path(params['test']).open('wb') as test_path:
                dump_svmlight_file(X[0:1], y[0:1], test_path)
            params['rlog'] = Path(tempdir).joinpath('learning.log').__str__()
            # params['out'] = Path(tempdir).joinpath('preds').__str__()
            params['save_model'] = Path(tempdir).joinpath('model').__str__()
            command = 'libFM ' + '-task c ' + ' '.join(['-' + k + ' ' + v for k, v in params.items()])
            if self.quiet:
                command += ' > /dev/null'
            path = 'PATH=' + os.environ['PATH']
            if self.libpath:
                path += ':' + self.libpath.__str__()
            command = ' '.join([path, command])
            # print(command)
            status = os.system(command)
            self.command = command
            if self.rlog:
                self.rlog_df = pd.read_table(params['rlog'], delim_whitespace=True)
            return self.load_model(params['save_model'])

    def load_model(self, path):
        """
        load model libfm format
        :param path: model location
        :return: self
        """
        with Path(path.__str__()).open('r') as f:
            saved_model_txt = [x.strip() for x in f.readlines()]
        idx = [(i, x.rstrip()) for i, x in enumerate(saved_model_txt) if x.startswith('#')]
        if self.k0 > 0:
            idx_bias = [i for i, x in idx if x.startswith('#global bias W0')][0] + 1
        if self.k1 > 0:
            idx_unary = slice([i for i, x in idx if x.startswith('#unary interactions Wj')][0] + 1, None)
        idx_pair = slice([i for i, x in idx if x.startswith('#pairwise interactions Vj,f')][0] + 1, None)
        if self.k1 > 0:
            idx_unary = slice(idx_unary.start, idx_pair.start - 1)
        if self.k0 > 0:
            self.W0_ = float(saved_model_txt[idx_bias])
        else:
            self.W0_ = 0
        if self.k1 > 0:
            self.Wj_ = np.array([float(x) for x in saved_model_txt[idx_unary]])
        if self.k2 > 0:
            self.Vjf_ = np.array([row.split() for row in saved_model_txt[idx_pair]]).astype(float)
        return self

    def _caluculate_score(self, X):
        # TODO: type chacking
        z = np.array([self.W0_] * X.shape[0])
        if self.k1 > 0:
            z += X.dot(self.Wj_)
        if self.k2 > 0:
            z += (X.dot(self.Vjf_) - X.power(2).dot(np.power(self.Vjf_, 2))).sum(axis=1) * .5
        return z

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(- self._caluculate_score(X)))
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return self.predict_proba(X) > .5

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))