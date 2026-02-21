#ifndef FANTASY_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define FANTASY_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace fantasy_infer
{
    class ReluLayer : public NonParamLayer
    {
    public:
        ReluLayer() : NonParamLayer("Relu") {}
        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

        static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &relu_layer);
    };
} // namespace fantasy_infer
#endif // FANTASY_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
