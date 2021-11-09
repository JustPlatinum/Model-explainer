from pdpbox import pdp
from matplotlib import pyplot as plt

import eli5
from eli5.sklearn import PermutationImportance

import warnings

import pandas as pd
import numpy as np
import shap

from abc import ABC, abstractmethod

import itertools

class BaseExplainer(ABC):
    """
    Базовый класс для интерпретатора.
    """

    @abstractmethod
    def fit(self):
        """
        Применение вычислительной функции к данным
        :returns: Интерпретатор с расчитанными значениями анализаторов
        """
        pass

    @abstractmethod
    def to_disc(self, path: str):
        """
        Сериализация модели отбора признаков на диск
        """
        pass

    @classmethod
    @abstractmethod
    def from_disc(cls, path: str):
        """
        Чтение модели отбора признаков с диска
        """
        pass

class PdExplainer(BaseExplainer):
    """Класс PdExplainer используется для интерпретации моделей машинного обучения.

    Основное применение - интерпретация моделей с помощью Partial Dependence.

    Note:
    Класс использует библиотеку PDP-Box(https://github.com/SauceCat/PDPbox). 
    Наиболее простая установка через conda-forge(https://github.com/conda-forge/pdpbox-feedstock).
    
    Attributes
    ----------
    model : None
        предобученная модель
    x : pd.DataFrame
        данные с признаками (Например: X, X_test, X_train и т.д.)
    model_features : list
        список признаков(названий колонок)  
    features : list
        список признаков(названий колонок) для которых будет вычисляться PartialDependence
    """

    def __init__(
        self,
        model,
        x: pd.DataFrame,
        model_features: list,
        features: list):
        """
        Args:
            model: предобученная модель
            x: данные с признаками (Например: X, X_test, X_train и т.д.)
            model_features: список признаков(названий колонок)
            features: список признаков(названий колонок) для которых будет вычисляться PartialDependence
        """
        self.model = model
        self.X = x
        self.model_features = model_features
        self.features=features

    def fit(self,fit_pairs=False) -> None:
        """
        Метод вычисляет значения параметров, которые используются для интерпретации Partial Dependence.
        :param fit_pairs: По умолчанию False. Если True, то PartialDependence будет расчитан не только для признаков в отдельности, но и для пар признаков.
        :return: Интерпретатор с расчитанными параметрами PartialDependence.
        """
        features = self.features
        pdp_feat_list=[]

        for feat in features:
            pdp_feat = pdp.pdp_isolate(model=self.model, dataset=self.X, model_features=self.model_features, feature=feat)
            pdp_feat_list.append(pdp_feat)
        
        self.pdp_feat = dict(zip(features, pdp_feat_list)) 
        print('Fitted features:')
        print(list(self.pdp_feat.keys()))

        if (len(self.features)>=2) & (fit_pairs==True):

            pdp_intr_list=[]
            features_pairs = list(itertools.combinations(features, 2))

            for pair in features_pairs:
                pdp_intr = pdp.pdp_interact(model=self.model, dataset=self.X, model_features=self.model_features, features=pair)
                pdp_intr_list.append(pdp_intr)
                    
            self.pdp_intr = dict(zip(features_pairs, pdp_intr_list))   
            print('Fitted pairs:')
            print(str(list(self.pdp_intr.keys()))) 


    def plot_pdp(self, feature_name: str) -> None:
        """
        Метод строит PDP для одного признака.
        :param feature_name: название признака(колонки)
        """
        isolate_output = self.pdp_feat.get(feature_name)
        fig, axes = pdp.pdp_plot(isolate_output,feature_name)

        return fig, axes
    
    def plot_pdp_pair(self,pair_name: tuple) -> None:
        """
        Метод строит PDP для пары признаков.
        :param pair_name: кортеж из двух названий признаков(колонок)
        """
        interact_output = self.pdp_intr.get(pair_name)
        fig, axes = pdp.pdp_interact_plot(interact_output,pair_name)

        return fig, axes    
        

    def to_disc(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def from_disc(cls, path: str) -> 'PdExplainer':
        explainer = joblib.load(path)
        return explainer


class ShapExplainer(BaseExplainer):
    """Класс ShapExplainer используется для интерпретации моделей машинного обучения.

    Основное применение - интерпретация моделей с помощью SHAPley values.
    
    Attributes
    ----------
    model : None
        предобученная модель
    x : pd.DataFrame
        данные с признаками (Например: X, X_test, X_train и т.д.)
    """

    def __init__(
        self,
        model,
        x: pd.DataFrame,
        ):
        """
        Args:
            model: предобученная модель
            x: данные с признаками (Например: X, X_test, X_train и т.д.)
        """
        self.model = model   
        self.X = x
 
    def fit(self):
        """
        Метод вычисляет значения SHAP, которые используются для интерпретации SHAP.
        """
        shap.initjs()
        model_explainer = shap.KernelExplainer(self.model.predict, shap.kmeans(self.X,10).data)
        
        shap_values = model_explainer.shap_values(self.X)
        expected_value = model_explainer.expected_value
        
        self.shap_values = shap_values
        self.expected_value = expected_value      

    def plot_shap_summary(self):
        """
        Метод строит график SHAP summary.
        """
        ax = shap.summary_plot(self.shap_values, self.X,feature_names=self.X.columns.to_list())
        return ax

    def plot_shap_importance(self):
        """
        Метод строит график SHAP importance.
        """
        ax = shap.summary_plot(self.shap_values, self.X, feature_names=self.X.columns.to_list(),plot_type="bar")
        return ax   

    def plot_shap_dependence(self,feature: str):
        """
        Метод строит график SHAP dependence для выбранного признака(колонки).
        :param feature: название признака(колонки) для которой будет построен график
        """
        ax = shap.dependence_plot(feature, self.shap_values, self.X)
        return ax    

    def plot_shap_force(self,idx):
        """
        Метод строит график SHAP force для выбранного наблюдения из датасета.
        :param idx: индекс наблюдения в датасете
        """
        ax = shap.force_plot(self.expected_value, self.shap_values[idx,:], self.X.iloc[idx,:],feature_names=self.X.columns.to_list())
        return ax    

    def to_disc(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def from_disc(cls, path: str) -> 'ShapExplainer':
        explainer = joblib.load(path)
        return explainer

# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(rf).fit(X_train, Y_train)
# eli5.show_weights(perm, feature_names=X.columns.tolist())


class PermImpExplainer(BaseExplainer):
    """Класс PermImpExplainer используется для интерпретации моделей машинного обучения.

    Основное применение - интерпретация моделей с помощью Permutation Importance.
    
    Attributes
    ----------
    model : None
        предобученная модель
    x : pd.DataFrame
        данные с признаками (Например: X, X_test, X_train и т.д.)
    y : pd.DataFrame
        данные с таргетом (Например: y, y_test, y_train и т.д.)
    """

    def __init__(
        self,
        model,
        x: pd.DataFrame,
        y: pd.DataFrame):
        """
        Args:
            model: предобученная модель
            x: данные с признаками (Например: X, X_test, X_train и т.д.)
            y: данные с таргетом (Например: y, y_test, y_train и т.д.)
        """
        self.model = model
        self.X = x
        self.y = y

    def fit(self) -> None:
        """
        Метод вычисляет значения параметров, которые используются для интерпретации Permutation Importance.
        """
        perm = PermutationImportance(self.model).fit(self.X, self.y)
        self.perm = perm


    def plot_PIweights(self) -> None:
        """
        Метод выводит значения Permutation Importance.
        """
        from IPython.display import display
        display(eli5.show_weights(self.perm, feature_names=self.X.columns.tolist()))       

    def to_disc(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def from_disc(cls, path: str) -> 'PermImpExplainer':
        explainer = joblib.load(path)
        return explainer