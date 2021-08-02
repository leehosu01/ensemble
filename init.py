#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:57:19 2021

@author: map
"""

from typing import List, Callable, Union
import numpy as np
import random
def ensemble_predict_function(
    model_predict_functions     : List[Callable],
    weights                     : np.ndarray):
    def _sub(dataset):
        result = None
        for predict, weight in zip(model_predict_functions, weights):
            pred = predict(dataset)
            if result is None: result = pred * weight
            else: result += pred * weight
        return result
    return _sub
def normalize_weight(weight):
    return weight / (1e-9 + weight.sum(axis = -2))
def random_search_(low, upp, target_function):
    weight = low + (upp - low) * random.random()
    return target_function(weight), weight
def random_search(low, upp, target_function, select_best, search_precision = 20):
    functional_space = [random_search_(low, upp, target_function) for _ in range(search_precision)]
    return select_best(functional_space)
def ternary_search(low, upp, target_function, select_best, search_precision = 20):
    def _update(new_param):
        nonlocal best_param
        best_param = select_best([new_param, best_param])
    low_value = target_function(low)
    upp_value = target_function(upp)
    best_param = low_value, low
    _update((upp_value, upp))
    for search_attempts in range(search_precision):
        midlow = (low * 2 + upp) / 3
        midupp = (low + upp * 2) / 3
        midlow_value = target_function(midlow)
        midupp_value = target_function(midupp)
        _update((midlow_value, midlow))
        _update((midupp_value, midupp))
        if min(midlow_value, midupp_value) > max(low_value, upp_value) + 1e-6:
            print(f"target space seems unstable")
            print(f"({low:.4e}, {low_value:.4e}), ({midlow:.4e}, {midlow_value:.4e}), ({midupp:.4e}, {midupp_value:.4e}), ({upp:.4e}, {upp_value:.4e})")
            _update(random_search(low, upp, target_function, select_best, search_precision - search_attempts))
            break
        elif midlow_value < midupp_value: upp = midupp
        else : low = midlow
        #print(f"({low:.4e}, {low_value:.4e}), ({midlow:.4e}, {midlow_value:.4e}), ({midupp:.4e}, {midupp_value:.4e}), ({upp:.4e}, {upp_value:.4e})")
            
    return best_param
def stacking_ensemble(
    order               :List,
    model_predicts      :np.ndarray,
    eval_function       :Callable,
    eval_method         :Callable = np.argmin,
    rate_underbound     :Union[float, Callable] = 0.25,
    rate_upperbound     :Union[float, Callable] = 4.00 ,
    search_method       :Union[str, Callable] = 'auto',
    search_precision    :int = 40,
    verbose = 1):
    def select_best(cases): return cases[eval_method(list(zip(*cases))[0])]
    def _update(new_param):
        nonlocal best_param
        if eval_method([new_param[0], best_param[0]]) == 0:
            if verbose: print(f"eval update: {best_param[0]} -> {new_param[0]}")
            best_param = new_param[0], new_param[1].copy()
    def metric_helper(weight):
        W_shape  = [(I if i == 0 else 1) for i, I in enumerate(model_predicts.shape)]
        ens_pred = (model_predicts * np.reshape(weight, tuple(W_shape))).sum(axis = 0)
        return eval_function(ens_pred)
    weight = np.ones((len(model_predicts), ), dtype = 'float32')
    weight = normalize_weight(weight)
    best_param   = eval_function(normalize_weight(weight)), weight
    for i, current_order in enumerate(order):
        low = rate_underbound if type(rate_underbound) == float else rate_underbound(i / len(order))
        upp = rate_upperbound if type(rate_upperbound) == float else rate_upperbound(i / len(order))
        def setting(rate):
            RES = weight.copy()
            RES[current_order] *= rate
            return normalize_weight(RES)
        def target_function(rate):
            return metric_helper(setting(rate))
            
        metric, multipler = search_method(
            low = low,
            upp = upp,
            select_best = select_best,
            target_function = target_function,
            search_precision= search_precision)
        weight = setting(multipler)
        if verbose: print(f"{i}:model_{current_order}, {multipler:1.5f} : eval = {metric}")
        _update((metric, weight))
    return best_param
def ensemble(
    model_predict_functions     : List[Callable],
    dataset      ,
    eval_function       :Callable,
    eval_method         :Callable = np.argmin,
    random_sample_count :int = 16,
    random_order_length :int = 256,
    rate_underbound     :Union[float, Callable] = 0.25,
    rate_upperbound     :Union[float, Callable] = 4.00 ,
    search_method       :Callable = ternary_search,
    search_precision    :int = 20,
    use_memmap          :bool = False,
    verbose = 0) -> np.ndarray:
    
    """
    

    Parameters
    ----------
    model_predict_functions : List[Callable]
        multiple model's prediction function.
        Such as, [model.predict for model in models]
    dataset : TYPE
        dataset, which are use for input to predict function.
        It will called multiple times, and have to return the same order in each call.
    eval_function : Callable
        If eval_method can be process your function's return type, we do not care return shape.
        The function must consider target value yourself.
    eval_method : Callable, optional
        while ensembling, this package will select the best evaluation weight.
        if A, B, C, D case are provided and you need to select C, then this function return 2.
        The default is np.argmin.
    random_sample_count : int, optional
        ensemble order sampling. The default is 16.
    random_order_length : int, optional
        ensemble order length. The default is 256.
    rate_underbound : Union[float, Callable], optional
        it is underbound of multiplication.
        if you provide function, we will provide progress of current ensembing. 0. <= progress < 1.
        you need to return float.
        The default is 0.25.
    rate_upperbound : Union[float, Callable], optional
        it is upperbound of multiplication.
        if you provide function, we will provide progress of current ensembing. 0. <= progress < 1.
        you need to return float.
        The default is 4.00.
    search_method : Callable, optional
        we provide 'ternary_search' and 'random_search'.
        The default is 'ternary_search'.
    search_precision : int, optional
        if ternary, weight will be determined in (2/3)**search_precision
        if random, weight will be determined in almost 1/search_precision
        The default is 20.
    use_memmap : bool, optional
        If you have limited memory, use True. The default is False.

    Returns
    -------
    return weight of ensembling.

    """
    
    best_param = None
    def _update(new_param):
        nonlocal best_param
        if best_param is None or eval_method([new_param[0], best_param[0]]) == 0:
            if verbose: print(f"eval update: {best_param[0]} -> {new_param[0]}")
            best_param = new_param[0], new_param[1].copy()
    
    model_predicts = np.array([predict_function(dataset) 
                               for predict_function in model_predict_functions])
    for trial in range(random_sample_count): 
        order = np.arange(random_order_length) % len(model_predict_functions)
        np.random.shuffle(order)
        new_param = stacking_ensemble( order = order.tolist(),
                                            model_predicts      = model_predicts,
                                            eval_function       = eval_function,
                                            eval_method         = eval_method,
                                            rate_underbound     = rate_underbound,
                                            rate_upperbound     = rate_upperbound,
                                            search_method       = search_method,
                                            search_precision    = search_precision,
                                            verbose = verbose)
        _update(new_param)
    return best_param[1]