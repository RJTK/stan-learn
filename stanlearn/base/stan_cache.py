import os
import pickle

from hashlib import md5
from copy import deepcopy

import pystan

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "../linear_regression/stan_models/")


class StanCacheMixin:
    def _get_file_hash(file_loc):
        with open(file_loc, "r") as f:
            hsh = md5(f.read().encode("utf-8")).hexdigest()
        return hsh

    def _get_stan_file_loc(name):
        return MODEL_DIR + name + ".stan"

    def _get_model_pkl_loc(name):
        return MODEL_DIR + name + ".pkl"

    def __init__(self):
        return

    def _setup_predict_kwargs(self, data, extra_kwargs):
        fit_kwargs = deepcopy(self.stan_fitting_kwargs)
        fit_kwargs.update(extra_kwargs)
        fit_kwargs["data"] = data
        return fit_kwargs

    def _get_name(self):
        return type(self).__name__

    def _load_compiled_models(self):
        stan_model = self._load_compiled_model(
            self._get_name() + "_model")
        predict_model = self._load_compiled_model(
            self._get_name() + "_predict")
        return stan_model, predict_model

    def _load_compiled_model(self, name):
        try:
            with open(MODEL_DIR + name + ".pkl", "rb") as f:
                _model = pickle.load(f)
                model = _model["model"]
                model_hsh = _model["md5"]
        except FileNotFoundError:
            model, model_hsh = self._compile_model(name)
        else:
            file_hsh = StanCacheMixin._get_file_hash(
                StanCacheMixin._get_stan_file_loc(name))
            if model_hsh != file_hsh:
                model, model_hsh = self._compile_model(name)
        _model = {"model": model, "md5": model_hsh}
        self._save_compiled_model(_model, name)
        return model

    def _compile_model(self, name):
        file_loc = StanCacheMixin._get_stan_file_loc(name)
        model_hsh = StanCacheMixin._get_file_hash(file_loc)
        model = pystan.StanModel(file=file_loc, model_name=name,
                                 include_paths=MODEL_DIR)
        return model, model_hsh

    def _save_compiled_model(self, model, name):
        file_loc = StanCacheMixin._get_model_pkl_loc(name)
        with open(file_loc, "wb") as f:
            pickle.dump(model, f)
        return
