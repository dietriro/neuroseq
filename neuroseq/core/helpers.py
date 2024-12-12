import multiprocessing as mp
import traceback
import numpy as np

from neuroseq.common.config import SYMBOLS


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            # ToDo: Figure out why the exception doesn't get carried back to top
            # print(e.with_traceback(None))
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def symbol_from_label(label, endpoint):
    return label.split('_')[1].split('>')[endpoint]


def id_to_symbol(index):
    return list(SYMBOLS.keys())[index]


def moving_average(a, n=4):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    if n > 1:
        return np.concatenate((moving_average(a[:n - 1], n=n - 1), ret), axis=0)
    else:
        return ret


def calculate_trace(start_value, t_start, t_end, t_events, decay_factor):
    trace_new = start_value
    for t_event in t_events:
        if not t_start < t_event < t_end:
            continue
        trace_new = trace_new * np.exp(-(t_event - t_start) / decay_factor) + 1.0
        t_start = t_event

    trace_new = trace_new * np.exp(-(t_end - t_start) / decay_factor)

    return trace_new


def psp_max_2_psc_max(psp_max, tau_m, tau_s, C_m):
    """Compute the PSC amplitude (pA) injected to get a certain PSP maximum (mV) for LIF with exponential PSCs

    Parameters
    ----------
    psp_max: float
             Maximum postsynaptic pontential
    tau_m:   float
             Membrane time constant (ms).
    tau_s:   float
             Synaptic time constant (ms).
    C_m:     float
             Membrane resistance (Gohm).

    Returns
    -------
    float
        PSC amplitude (pA).
    """
    R_m = tau_m / C_m
    return psp_max / (
            R_m * tau_s / (tau_s - tau_m) * (
                (tau_m / tau_s) ** (-tau_m / (tau_m - tau_s)) -
                (tau_m / tau_s) ** (-tau_s / (tau_m - tau_s))
            )
    )


class Latex:
    @staticmethod
    def print_title(title, num_cols=3):
        print("\\hline")
        print(f"\\multicolumn{{{num_cols}}}{{c}}{{\\bfseries {title}}} \\\\ ")
        print("\\hline")

    @staticmethod
    def print_row(row, order=None, comment=None, hline=False):
        if order is not None:
            new_row = [row[i] for i in order]
        else:
            new_row = row
        output = ' & '.join(new_row) + ' \\\\'
        if comment is not None:
            output += f'  % {comment}'
        print(output)
        if hline:
            print('\\hline')
