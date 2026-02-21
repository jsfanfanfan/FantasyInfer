#ifndef FANTASY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define FANTASY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_op.hpp"

namespace fantasy_infer
{
    class LayerRegisterer
    {
    public:
        typedef ParseParameterAttrStatus (*Creator)(
            /**
             * 不同算子的实例化函数需要接受两个参数
             * @param op 保存了初始化 Layer 信息的算子 RuntimeOperator
             * @param layer 需要被初始化的 Layer
             * @return ParseParameterAttrStatus类型的状态值
             */
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &layer);
        
        // 全局注册表(map)，key 是算子的类型(string)
        // value 是 creator，是一个函数指针，指向算子的初始化函数
        typedef std::map<std::string, Creator> CreateRegistry;

    public:
        /**
         * 向注册表注册算子
         * @param layer_type 算子的类型
         * @param creator 需要注册算子的注册表
         */
        static void RegisterCreator(const std::string &layer_type,
                                    const Creator &creator);

        /**
         * 通过算子参数 op 来初始化 Layer
         * @param op 保存了初始化 Layer 信息的算子
         * @return 初始化后的 Layer
         */
        static std::shared_ptr<Layer> CreateLayer(
            const std::shared_ptr<RuntimeOperator> &op);

        /**
         * 返回算子的注册表
         * @return 算子的注册表
         */
        static CreateRegistry &Registry();

        /**
         * 返回所有已被注册算子的类型
         * @return 注册算子的类型列表
         */
        static std::vector<std::string> layer_types();
    };

    class LayerRegistererWrapper
    {
    public:
        LayerRegistererWrapper(const std::string &layer_type,
                               const LayerRegisterer::Creator &creator)
        {
            LayerRegisterer::RegisterCreator(layer_type, creator);
        }
    };

} // namespace fantasy_infer

#endif // FANTASY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
