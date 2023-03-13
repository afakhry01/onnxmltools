# SPDX-License-Identifier: Apache-2.0

'''
Mapping and utility functions for Name to Spark ML operators
'''

from pyspark.ml import Transformer, Estimator
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import ChiSqSelectorModel
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.feature import DCT
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDFModel
from pyspark.ml.feature import ImputerModel
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import MaxAbsScalerModel
from pyspark.ml.feature import MinHashLSHModel
from pyspark.ml.feature import MinMaxScalerModel
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import OneHotEncoderModel
from pyspark.ml.feature import PCAModel
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexerModel
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import Word2VecModel

from pyspark.ml.classification import LinearSVCModel, RandomForestClassificationModel, GBTClassificationModel, \
    MultilayerPerceptronClassificationModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.classification import OneVsRestModel

from pyspark.ml.regression import AFTSurvivalRegressionModel, DecisionTreeRegressionModel, RandomForestRegressionModel
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.regression import GeneralizedLinearRegressionModel
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.ml.regression import LinearRegressionModel

from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import LDA

from ..common._registration import register_converter, register_shape_calculator

def build_sparkml_operator_name_map():
    res = {k: "pyspark.ml.feature." + k.__name__ for k in [
        Binarizer, BucketedRandomProjectionLSHModel, Bucketizer,
        ChiSqSelectorModel, CountVectorizerModel, DCT, ElementwiseProduct, HashingTF, IDFModel, ImputerModel,
        IndexToString, MaxAbsScalerModel, MinHashLSHModel, MinMaxScalerModel, NGram, Normalizer, OneHotEncoderModel,
        PCAModel, PolynomialExpansion, QuantileDiscretizer, RegexTokenizer,
        StandardScalerModel, StopWordsRemover, StringIndexerModel, Tokenizer, VectorAssembler, VectorIndexerModel,
        VectorSlicer, Word2VecModel
    ]}
    res.update({k: "pyspark.ml.classification." + k.__name__ for k in [
        LinearSVCModel, LogisticRegressionModel, DecisionTreeClassificationModel, GBTClassificationModel,
        RandomForestClassificationModel, NaiveBayesModel, MultilayerPerceptronClassificationModel, OneVsRestModel
    ]})
    res.update({k: "pyspark.ml.regression." + k.__name__ for k in [
        AFTSurvivalRegressionModel, DecisionTreeRegressionModel, GBTRegressionModel, GBTRegressionModel,
        GeneralizedLinearRegressionModel, IsotonicRegressionModel, LinearRegressionModel, RandomForestRegressionModel
    ]})
    res.update({k: "pyspark.ml.clustering." + k.__name__ for k in [
        KMeansModel
    ]})
    return res


sparkml_operator_name_map = build_sparkml_operator_name_map()


def get_sparkml_operator_name(model_type):
    '''
    Get operator name of the input argument

    :param model_type:  A spark-ml object (LinearRegression, StringIndexer, ...)
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if not issubclass(model_type, Transformer):
        if issubclass(model_type, Estimator):
            raise ValueError("Estimator must be fitted before being converted to ONNX")
        else:
            raise ValueError("Unknown model type: {}".format(model_type))

    return sparkml_operator_name_map[model_type]


def update_registered_converter(model, alias, shape_fct, convert_fct,
                                overwrite=True, parser=None, options=None):
    """
    Registers or updates a converter for a new model so that
    it can be converted when inserted in a *scikit-learn* pipeline.

    :param model: model class
    :param alias: alias used to register the model
    :param shape_fct: function which checks or modifies the expected
        outputs, this function should be fast so that the whole graph
        can be computed followed by the conversion of each model,
        parallelized or not
    :param convert_fct: function which converts a model
    :param overwrite: False to raise exception if a converter
        already exists
    :param parser: overwrites the parser as well if not empty
    :param options: registered options for this converter

    The alias is usually the library name followed by the model name.
    Example:

    ::

        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.operator_converters.RandomForest import convert_sklearn_random_forest_classifier
        from skl2onnx import update_registered_converter
        update_registered_converter(
                SGDClassifier, 'SklearnLinearClassifier',
                calculate_linear_classifier_output_shapes,
                convert_sklearn_random_forest_classifier,
                options={'zipmap': [True, False, 'columns'],
                         'output_class_labels': [False, True],
                         'raw_scores': [True, False]})

    The function does not update the parser if not specified except if
    option `'zipmap'` is added to the list. Every classifier
    must declare this option to let the default parser
    automatically handle that option.
    """  # noqa
    # if (not overwrite and model in sparkml_operator_name_map
    #         and alias != sparkml_operator_name_map[model]):
    #     warnings.warn("Model '{0}' was already registered under alias "
    #                   "'{1}'.".format(model, sparkml_operator_name_map[model]))
    sparkml_operator_name_map[model] = alias
    register_converter(alias, convert_fct, overwrite=overwrite)
    register_shape_calculator(alias, shape_fct, overwrite=overwrite)
