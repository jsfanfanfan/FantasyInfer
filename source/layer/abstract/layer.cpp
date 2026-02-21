#include "layer/abstract/layer.hpp"
namespace fantasy_infer
{

    const std::vector<std::shared_ptr<Tensor<float>>> &Layer::weights() const
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    const std::vector<std::shared_ptr<Tensor<float>>> &Layer::bias() const
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::vector<float> &bias)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(const std::vector<float> &weights)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    void Layer::set_weights(
        const std::vector<std::shared_ptr<Tensor<float>>> &weights)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    InferStatus Layer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
        std::vector<std::shared_ptr<Tensor<float>>> &outputs)
    {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
    }

    // 不带参数的前向传播 Forward 函数
    // 作用是准备输入和输出数据
    // 并使用这些数据调用每个派生类算子中各自实现实现的计算过程
    InferStatus Layer::Forward()
    {
        LOG_IF(FATAL, this->runtime_operator_.expired())
            << "Runtime operator is expired or nullptr";
        // 获取 Layer 对应的 RuntimeOperator
        const auto &runtime_operator = this->runtime_operator_.lock();
        // 准备 layer 计算所需要的输入数据
        const std::vector<std::shared_ptr<RuntimeOperand>> &input_operand_datas =
            runtime_operator->input_operands_seq;
        // layer的输入
        std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
        // 输入操作数可能有多个来源，多个前置节点
        for (const auto &input_operand_data : input_operand_datas)
        {
            for (const auto &input_data : input_operand_data->datas)
            {
                layer_input_datas.push_back(input_data);
            }
        }
        // output 存放对应的结果，是一个预申请好的空间
        const std::shared_ptr<RuntimeOperand> &output_operand_datas =
            runtime_operator->output_operands;

        CHECK(!layer_input_datas.empty())
            << runtime_operator->name << " Layer input data is empty";
        CHECK(output_operand_datas != nullptr && !output_operand_datas->datas.empty())
            << "Layer output data is empty";
        // 执行 operator 当中的 layer 计算过程
        // layer 的计算结果存放在 current_op->output_operands->datas 中
        // 调用子类各自带参数的 Forward 实现
        InferStatus status = runtime_operator->layer->Forward(
            layer_input_datas, output_operand_datas->datas);
        return status;
    }

    void Layer::set_runtime_operator(
        const std::shared_ptr<RuntimeOperator> &runtime_operator)
    {
        CHECK(runtime_operator != nullptr);
        this->runtime_operator_ = runtime_operator;
    }

} // namespace fantasy_infer