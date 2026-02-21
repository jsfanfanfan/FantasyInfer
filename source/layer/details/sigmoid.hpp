#ifndef FANTASY_INFER_SOURCE_LAYER_BINOCULAR_SIGMOID_HPP_
#define FANTASY_INFER_SOURCE_LAYER_BINOCULAR_SIGMOID_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace fantasy_infer
{
    class SigmoidLayer : public NonParamLayer
    {
    public:
        SigmoidLayer() : NonParamLayer("Sigmoid") {}
        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &sigmoid_layer);
    };
} // namespace fantasy_infer
#endif // FANTASY_INFER_SOURCE_LAYER_BINOCULAR_SIGMOID_HPP_
