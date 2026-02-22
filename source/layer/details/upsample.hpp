#ifndef FANTASY_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#define FANTASY_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace fantasy_infer {
enum class UpSampleMode {
  kModeNearest = 0,  // 目前上采样层只支持邻近采样
};

class UpSampleLayer : public NonParamLayer {
 public:
  explicit UpSampleLayer(float scale_h, float scale_w,
                         UpSampleMode mode = UpSampleMode::kModeNearest);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& upsample_layer);

 private:
  float scale_h_ = 1.f;
  float scale_w_ = 1.f;
  UpSampleMode mode_ = UpSampleMode::kModeNearest;
};
}  // namespace fantasy_infer
#endif  // FANTASY_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
