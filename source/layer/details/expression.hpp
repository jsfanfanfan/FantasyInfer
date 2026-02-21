#ifndef FANTASY_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#define FANTASY_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "parser/parser_expression.hpp"

namespace fantasy_infer {
class ExpressionLayer : public NonParamLayer {
 public:
  explicit ExpressionLayer( std::string statement);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& expression_layer);

 private:
  std::string statement_;
  std::unique_ptr<ExpressionParser> parser_;
};
}  // namespace fantasy_infer
#endif  // fantasy_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
