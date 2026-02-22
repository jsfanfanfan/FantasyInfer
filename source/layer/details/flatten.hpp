#ifndef FANTASY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define FANTASY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace fantasy_infer {
class FlattenLayer : public NonParamLayer {
 public:
  explicit FlattenLayer(int start_dim, int end_dim);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus CreateInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& flatten_layer);

 private:
  int start_dim_ = 0;
  int end_dim_ = 0;
};
}  // namespace fantasy_infer
#endif  // FANTASY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
