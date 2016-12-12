from keras.models import Sequential
from keras.layers import TimeDistributed, Dense
from keras.layers.recurrent import PLSTM, LSTM
from keras.callbacks import EarlyStopping, Callback

from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
import time

import numpy as np
import pylab as py
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


class PLSTMMon(Callback):
    def on_train_begin(self, logs={}):
        self.history = defaultdict(list)

    def on_epoch_end(self, epoch, logs={}):
        for f in ('s', 'tau', 'r_on'):
            val = getattr(self.model.get_layer('plstm'), f).eval().flatten()
            self.history[f].append(val)


def curve(t, theta, phi, sigma):
    return np.sin(theta*t + phi) + sigma * 2 * (np.random.random(len(t))-0.5)


def make_data(curves, t, chunk_size=10):
    data_samples = []
    for i, (theta, phi, sigma) in enumerate(curves):
        d = curve(t, theta, phi, sigma)
        data_samples.extend(zip(d, t, (i,)*len(t)))
    data_samples.sort(key=lambda x: x[1])
    max_index = (len(data_samples)//chunk_size) * chunk_size
    for i in range(0, max_index, chunk_size):
        yield data_samples[i:i+chunk_size]


def plot_history(hist, filename=None):
    f, axs = py.subplots(len(hist), sharex=True)
    for ax, (k, d) in zip(axs, hist.items()):
        d = np.asarray(d)
        if len(d.shape) == 2:
            data_iter = d.T
        elif len(d.shape) == 1:
            data_iter = [d]
        for line in data_iter:
            ax.plot(line)
        ax.set_ylabel(k)
    if filename:
        f.savefig(filename)


def plot_histories(hist, filename=None):
    keys = [k for k in hist[0].keys()
            if len(np.asarray(hist[0][k]).shape) == 1]
    f, axs = py.subplots(len(keys), sharex=True)
    py.title(filename)
    for ax, k in zip(axs, keys):
        if len(np.asarray(hist[0][k]).shape) != 1:
            continue
        d = np.asarray([h[k] for h in hist])
        x = list(range(d.shape[1]))
        lower = np.percentile(d, 0.25, axis=0)
        upper = np.percentile(d, 0.75, axis=0)
        middle = np.percentile(d, 0.5, axis=0)
        ax.fill_between(x, lower, upper, alpha=0.5)
        ax.plot(x, middle)
        ax.set_ylabel(k)
    if filename:
        f.savefig(filename)


def plot_data(X, y, curves, filename=None):
    f, ax = py.subplots()
    t = X[:, :, 1]
    for theta, phi, sigma in curves:
        x = curve(t.flatten(), theta, phi, 0)
        py.plot(t.flatten(), x)
    c = [(0, 1, 0) if i else (1, 0, 0) for i in y.flatten()]
    ax.scatter(t, X[:, :, 0], c=c)
    if filename:
        f.savefig(filename)


def plot_k(hist, T, indexes=None, alpha=0.005, filename=None):
    indexes = indexes or range(len(hist['r_on']))
    f, axs = py.subplots(len(indexes), sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    for ax, idx in zip(axs, indexes):
        r_on = hist['r_on'][idx]
        tau = hist['tau'][idx]
        s = hist['s'][idx]
        phi_t = ((T[:, np.newaxis].repeat(s.shape[0], -1) - s) % tau) / tau
        k = np.zeros_like(phi_t)
        for i in range(phi_t.shape[0]):
            for j in range(phi_t.shape[1]):
                if r_on[j] / 2 < phi_t[i, j] < r_on[j]:
                    k[i, j] = 2 - 2 * phi_t[i, j] / r_on[j]
                elif phi_t[i, j] < r_on[j] / 2:
                    k[i, j] = 2 * phi_t[i, j] / r_on[j]
                else:
                    k[i, j] = alpha * phi_t[i, j]
        ax.plot(T, k)
        ax.set_ylabel(idx)
    if filename:
        f.savefig(filename)


def random_interval_range(start, end, rand_small, rand_large,
                          rand=np.random.random):
    values = [start]
    while True:
        propose = rand() * (rand_large - rand_small) + rand_small
        if values[-1] + propose > end:
            return np.asarray(values)
        values.append(values[-1] + propose)


def set_random():
    np.random.seed(np.int32(time.monotonic() * 1e10) % 4294967295)


def model_lstm(X, y, nb_epoch=100, patience=2):
    set_random()
    lstm = Sequential()
    lstm.add(LSTM(16, input_shape=(None, 2), name='lstm',
                  return_sequences=True))
    lstm.add(TimeDistributed(Dense(1, activation='sigmoid', init='normal')))
    lstm.compile('adam', metrics=['accuracy'], loss='binary_crossentropy')

    es = EarlyStopping(patience=patience, monitor='val_loss')
    hist = lstm.fit(X, y, nb_epoch=nb_epoch, verbose=0, validation_split=0.1,
                    )#callbacks=[es])
    print("\t".join(["{}: {}".format(k, h[-1])
                     for k, h in hist.history.items()]))
    return hist.history


def model_plstm(X, y, nb_epoch=100, patience=2):
    set_random()
    plstm = Sequential()
    plstm.add(PLSTM(16, input_shape=(None, 2), name='plstm',
                    return_sequences=True, consume_less='mem'))
    plstm.add(TimeDistributed(Dense(1, activation='sigmoid', init='normal')))

    print("Compiling")
    plstm.compile('adam', metrics=['accuracy'], loss='binary_crossentropy')

    print("Fitting")
    plstmmon = PLSTMMon()
    es = EarlyStopping(patience=patience, monitor='val_loss')
    history = plstm.fit(X, y, nb_epoch=nb_epoch, verbose=0,
                        validation_split=0.1,
                        callbacks=[plstmmon,])# es])
    print("\t".join(["{}: {}".format(k, h[-1])
                     for k, h in history.history.items()]))
    return {**history.history, **plstmmon.history}


def multiple_experiments(experiment, nb_experiments, curves, t, **kwargs):
    histories = []
    with Pool(processes=cpu_count()) as pool:
        datas = []
        for i in range(nb_experiments):
            data = np.array(list(make_data(
                curves,
                t(),
            )))
            X, y = data[:, :, :-1], data[:, :, -1:]
            datas.append((X, y))

        print("Running experiments: ", experiment.__name__)
        histories = pool.starmap(partial(experiment, **kwargs), datas)
    return histories


if __name__ == "__main__":
    nb_epoch = 600
    patience = 50
    nb_experiments = 48

    curves = ((1.000, 3.14, 0.25),
              (1.000, 0.00, 0.25))
    t = lambda: random_interval_range(16, 32, 0.01, 1)
    # t= [np.arange(16, 32, 0.1)]

    params = dict(nb_epoch=nb_epoch, patience=patience)

    histories_lstm = multiple_experiments(model_lstm, nb_experiments,
                                          curves, t, **params)
    plot_histories(histories_lstm, filename='histories_lstm.png')

    histories_plstm = multiple_experiments(model_plstm, nb_experiments,
                                           curves, t, **params)
    plot_histories(histories_plstm, filename='histories_plstm.png')

    history_lstm_best = min(histories_lstm, key=lambda x: x['val_loss'][-1])
    history_plstm_best = min(histories_plstm, key=lambda x: x['val_loss'][-1])
    plot_history(history_lstm_best, filename='history_lstm_best.png')
    plot_history(history_plstm_best, filename='history_plstm_best.png')

    plot_k(history_plstm_best, np.arange(np.pi, 5*np.pi, 0.1),
           indexes=[0, int(len(history_plstm_best['loss'])/2), -1],
           filename='k_gates.png')
    py.show()
