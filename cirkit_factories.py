from cirkit.utils.scope import Scope
from cirkit.symbolic.parameters import LogSoftmaxParameter, SoftmaxParameter, ExpParameter, Parameter, TensorParameter
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, HadamardLayer, MixingLayer
from cirkit.symbolic.initializers import NormalInitializer


def categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int
) -> CategoricalLayer:
    return CategoricalLayer(
        scope, num_units, num_channels, num_categories=2,
        probs_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
        )
    )


def hadamard_layer_factory(
    scope: Scope, num_input_units: int, arity: int
) -> HadamardLayer:
    return HadamardLayer(scope, num_input_units, arity)


def dense_layer_factory(
    scope: Scope,
    num_input_units: int,
    num_output_units: int
) -> DenseLayer:
    return DenseLayer(
        scope, num_input_units, num_output_units,
        weight_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
        )
    )
    
def mixing_layer_factory(
    scope: Scope, num_units: int, arity: int
) -> MixingLayer:
    return MixingLayer(
        scope, num_units, arity,
        weight_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1))
        )
    )