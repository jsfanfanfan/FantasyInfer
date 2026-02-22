#ifndef FANTASY_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define FANTASY_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace fantasy_infer {
class SoftmaxLayer : public NonParamLayer {
 public:
  explicit SoftmaxLayer(int dim = -1);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus CreateInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& softmax_layer);

 private:
  int softmax_dim_ = -1;
};
}  // namespace fantasy_infer

#endif  // FANTASY_INFER_SOURCE_LAYER_SOFTMAX_HPP_
