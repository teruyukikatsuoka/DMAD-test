#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
import sicore
import mpmath as mp

# significance level
ALPHA = 0.05


class PararellExperiment(metaclass=ABCMeta):
    def __init__(self, num_iter: int, num_worker: int):
        self.num_iter = num_iter
        self.num_worker = num_worker

    @abstractmethod
    def iter_experiment(self, data) -> tuple:
        """run experiment for each iteration.

        Args:
            data : data for experiment
        Return:
            tuple: result of experiment
        """
        pass

    def experiment(
        self,
        dataset: list,
    ) -> list:
        """
        Args:
            dataset (tuple): Dataset whose size must be equal to num_iter for experiment.

        Returns:
            list: List of results.
        """

        # do experiments with  multiprocessing
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, enumerate(dataset)), total=self.num_iter)
            )

        return results
    
    @abstractmethod
    def run_experiment(self):
        pass


def generate_images_classification(shape, num_iter, signal_1, signal_2) -> tuple:
    """generate a dataset

    Args:
        shape (Tuple): the image shape of the input
        num_iter (int): the number of samples to generate
        signal_1 (float): the signal of the normal region
        signal_2 (float): the signal of the abnormal region
    """

    X_test = []
    Y_test = []

    a = shape[0] // 2

    for i in range(num_iter):
        X = np.random.normal(signal_1, 1, shape)
        Y = np.zeros(shape)
        abnormal_x = np.random.randint(0, shape[0] - a + 1)
        abnormal_y = np.random.randint(0, shape[1] - a + 1)
        X[
            abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a, 0
        ] = np.random.normal(signal_2, 1, (a, a))
        Y[abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a, 0] = 1

        X_test.append(X)
        Y_test.append(Y)

    return X_test, Y_test

def chi2_cdf_mpmath(x, df):
    """
    CDF of a chi-squared distribution.
    Args:
        x (float): Return the value at `x`.
        df (float): Degree of freedom.
    Returns:
        float: CDF value at `x`.
    """

    return mp.gammainc(df / 2, a=0, b=x / 2, regularized=True)



def calc_p_value_chi2(chi, intervals, dof):
    mp.dps = 1000

    numerator = 0
    denominator = 0

    for interval in intervals:
        denominator += chi2_cdf_mpmath(interval[1] ** 2, dof) - chi2_cdf_mpmath(
            interval[0] ** 2, dof
        )
        if chi < interval[0]:
            numerator += chi2_cdf_mpmath(interval[1] ** 2, dof) - chi2_cdf_mpmath(
                interval[0] ** 2, dof
            )
        elif interval[0] <= chi <= interval[1]:
            numerator += chi2_cdf_mpmath(interval[1] ** 2, dof) - chi2_cdf_mpmath(
                chi**2, dof
            )

    if denominator == 0:
        raise CantCalcPvalueError(chi, intervals)

    return numerator / denominator


def p_value_norm_abs(z, intervals, std):
    numerator = 0
    denominator = 0

    intervals = sicore.intervals.union_all(intervals, tol=1e-6)

    normalized_z = z / std
    normalized_intervals = [interval / std for interval in intervals]

    for interval in normalized_intervals:
        l = interval[0]
        u = interval[1]

        denominator = denominator + mp.ncdf(u) - mp.ncdf(l)

        # -abs() part
        if u < 0:
            if l <= -abs(normalized_z) and -abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(-abs(normalized_z)) - mp.ncdf(l)
            elif u <= -abs(normalized_z):
                numerator = numerator + mp.ncdf(u) - mp.ncdf(l)
        elif l <= 0 and 0 <= u:
            if l <= -abs(normalized_z):
                numerator = numerator + mp.ncdf(-abs(normalized_z)) - mp.ncdf(l)
            if 0 <= abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(u) - mp.ncdf(abs(normalized_z))
        elif 0 < l:
            if abs(normalized_z) <= l:
                numerator = numerator + mp.ncdf(u) - mp.ncdf(l)
            elif l <= abs(normalized_z) and abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(u) - mp.ncdf(abs(normalized_z))

    if denominator == 0:
        raise CantCalcPvalueError(normalized_z, normalized_intervals)

    return numerator / denominator


def breakpoint_to_interval(breakpoint_list: np.ndarray) -> list:
    """the function which convert the breakpoint_list that is [L1,U1,L2,U2,...] to the list which is [[L1,U1],[L2,U2]]

    Returns
        intervals
    """
    n = breakpoint_list.shape[0] // 2
    intervals = []
    for i in range(n):
        lower = breakpoint_list[2 * i]
        upper = breakpoint_list[2 * i + 1]
        intervals.append([lower.numpy(), upper.numpy()])

    return intervals
