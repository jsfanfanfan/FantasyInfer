#ifndef FANTASY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#define FANTASY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace fantasy_infer {
class CatLayer : public NonParamLayer {
 public:
  explicit CatLayer(int dim);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& cat_layer);

 private:
  int32_t dim_ = 0;
};
}  // namespace fantasy_infer
#endif  // FANTASY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
