#ifndef FANTASY_INFER_SOURCE_LAYER_ADAPTIVE_AVGPOOLING_HPP_
#define FANTASY_INFER_SOURCE_LAYER_ADAPTIVE_AVGPOOLING_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace fantasy_infer {
class AdaptiveAveragePoolingLayer : public NonParamLayer {
 public:
  explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus CreateInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& avg_layer);

 private:
  uint32_t output_h_ = 0;
  uint32_t output_w_ = 0;
};
}  // namespace fantasy_infer
#endif  // FANTASY_INFER_SOURCE_LAYER_ADAPTIVE_AVGPOOLING_HPP_
