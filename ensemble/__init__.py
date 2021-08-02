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
    weights                     : np.ndarray,
    use_advanced_feature        : bool = False,
    less_memory                 : bool = False):
    def _sub(dataset):
        result = None
        if use_advanced_feature == False:
            for predict, weight in zip(model_predict_functions, weights_):
                pred = predict(dataset)
                if result is None: result = pred * weight
                else: result += pred * weight
        else:
            model_predicts = np.array([predict_function(dataset) for predict_function in model_predict_functions])
            model_predicts = np.concatenate([model_predicts, np.sort(model_predicts, axis = 0)], axis = 0)
            
            W_shape  = [(I if i == 0 else 1) for i, I in enumerate(model_predicts.shape)]
            result = (model_predicts * np.reshape(weights_, tuple(W_shape))).sum(axis = 0)

        return result
    weights_ = weights.copy()
    return _sub
def normalize_weights(weights):
    return weights / (1e-9 + weights.sum(axis = 0))
def random_search_(low, upp, target_function):
    rate = low + (upp - low) * random.random()
    return target_function(rate), rate
def random_search(low, upp, target_function, select_best, is_better, search_precision = 20, verbose = 0):
    functional_space = [random_search_(low, upp, target_function) for _ in range(search_precision)]
    return select_best(functional_space)
def ternary_search(low, upp, target_function, select_best, is_better, search_precision = 20, verbose = 0):
    def _update(new_param):
        nonlocal best_param
        best_param = select_best([new_param, best_param])
    low_value = target_function(low)
    upp_value = target_function(upp)
    best_param = low_value, low
    _update((upp_value, upp))
    
    worst_side = [(low_value, low), (upp_value, upp)][is_better((low_value, low), (upp_value, upp))]    
    for search_attempts in range(search_precision):
        midlow = (low * 2 + upp) / 3
        midupp = (low + upp * 2) / 3
        midlow_value = target_function(midlow)
        midupp_value = target_function(midupp)
        #_update((midlow_value, midlow))
        #_update((midupp_value, midupp))
        
        best_mid = [(midlow_value, midlow), (midupp_value, midupp)][::-1][is_better((midlow_value, midlow), (midupp_value, midupp))]
        _update(best_mid)
        if not is_better(best_mid, worst_side): # In the same case, the condition does not apply
            if verbose  > 4:
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
    search_method       :Callable = 'auto',
    search_precision    :int = 40,
    less_memory         :bool = False,
    verbose = 1):
    def select_best(cases): return cases[eval_method(list(zip(*cases))[0])]
    def is_better(case1, case2): return bool(eval_method([case1[0], case2[0]]) == 0)
    def _update(new_param):
        nonlocal best_param
        if eval_method([best_param[0], new_param[0]]) == 1:
            if verbose > 1: print(f"eval update: {best_param[0]} -> {new_param[0]}")
            best_param = new_param[0], new_param[1].copy()
    def metric_helper(weights):
        W_shape  = [(I if i == 0 else 1) for i, I in enumerate(model_predicts.shape)]
        ens_pred = (model_predicts * np.reshape(weights, tuple(W_shape))).sum(axis = 0)
        return eval_function(ens_pred)
    weights = np.ones((len(model_predicts), ), dtype = 'float32')
    weights = normalize_weights(weights)
    best_param   = metric_helper(weights), weights
    for i, current_order in enumerate(order):
        low = rate_underbound if type(rate_underbound) == float else rate_underbound(i / len(order))
        upp = rate_upperbound if type(rate_upperbound) == float else rate_upperbound(i / len(order))
        def setting(rate):
            RES = weights.copy()
            RES[current_order] *= rate
            return normalize_weights(RES)
        def target_function(rate):
            #less memory 사용가능, rates를 받아서 쿼리는 따로넣어도 ens pred는 동시에 생성가능
            return metric_helper(setting(rate))
            
        metric, multipler = search_method(
            low = low,
            upp = upp,
            select_best = select_best,
            is_better = is_better,
            target_function = target_function,
            search_precision= search_precision,
            verbose = verbose)
        weights = setting(multipler)
        if verbose > 2: print(f"{i}:model_{current_order}, {multipler:1.5f} : eval = {metric}")
        _update((metric, weights))
    return best_param
def ensemble(
    model_predict_functions     : List[Callable],
    dataset      ,
    eval_function       :Callable,
    eval_method         :Callable = np.argmin,
    random_sample_count :int = 8,
    random_order_length :int = 512,
    rate_underbound     :Union[float, Callable] = 0.10,
    rate_upperbound     :Union[float, Callable] = 2.00 ,
    search_method       :Callable = ternary_search,
    search_precision    :int = 10,
    use_advanced_feature:bool = False,
    less_memory         :bool = False,
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
        ensemble order sampling. The default is 8.
    random_order_length : int, optional
        ensemble order length. The default is 512.
    rate_underbound : Union[float, Callable], optional
        it is underbound of multiplication.
        if you provide function, we will provide progress of current ensembing. 0. <= progress < 1.
        you need to return float.
        The default is 0.10.
    rate_upperbound : Union[float, Callable], optional
        it is upperbound of multiplication.
        if you provide function, we will provide progress of current ensembing. 0. <= progress < 1.
        you need to return float.
        The default is 2.00.
    search_method : Callable, optional
        we provide 'ternary_search' and 'random_search'.
        The default is 'ternary_search'.
    search_precision : int, optional
        if ternary, weight will be determined in (2/3)**search_precision
        if random, weight will be determined in almost 1/search_precision
        The default is 10.
    use_advanced_feature : bool, optional
        There are cases where experimentally strong performance is shown, but this is not always the case.
        The default is False.
    less_memory : bool, optional
        If you have limited memory, use True.
        Algorithm will much slower
        Still not implemented
        The default is False.

    Returns
    -------
    return weight of ensembling.

    """
    
    best_param = None, None
    def _update(new_param, trial):
        nonlocal best_param
        if best_param[0] is None or eval_method([best_param[0], new_param[0]]) == 1:
            if verbose > 0: print(f"eval update at trial {trial:04d}: {new_param[0]}")
            best_param = new_param[0], new_param[1].copy()
    #less memory 사용가능
    model_predicts = np.array([predict_function(dataset) 
                               for predict_function in model_predict_functions])
    if use_advanced_feature:
        model_predicts = np.concatenate([model_predicts, np.sort(model_predicts, axis = 0)], axis = 0)
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
                                            less_memory         = less_memory,
                                            verbose             = verbose)
        _update(new_param, trial)
    return best_param[1]