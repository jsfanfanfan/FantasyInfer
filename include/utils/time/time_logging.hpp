#ifndef FANTASY_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
#define FANTASY_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
namespace fantasy_infer {
namespace utils {
using Time = std::chrono::steady_clock;

// 每个类型的层执行时间消耗
struct LayerTimeState {
  explicit LayerTimeState(long duration_time, std::string layer_name,
                          std::string layer_type)
      : duration_time_(duration_time),
        layer_name_(std::move(layer_name)),
        layer_type_(std::move(layer_type)) {}

  long duration_time_;      // 时间消耗
  std::mutex time_mutex_;   // 修改duration_time_时，所需要获取的锁
  std::string layer_name_;  // 层的名称
  std::string layer_type_;  // 层的类型
};

// 各类型层的时间消耗记录map类型
using LayerTimeStatesCollector =
    std::map<std::string, std::shared_ptr<LayerTimeState>>;

// 各类型层的时间消耗记录map指针类型
using PtrLayerTimeStatesCollector = std::shared_ptr<LayerTimeStatesCollector>;

class LayerTimeStatesSingleton {
 public:
  LayerTimeStatesSingleton() = default;

  /**
   * 初始化各类型层的时间消耗记录map
   */
  static void LayerTimeStatesCollectorInit();

  /**
   * 初始化各类型层的时间消耗记录map
   * @return 记录各类型层的时间消耗map
   */
  static PtrLayerTimeStatesCollector SingletonInstance();

 private:
  // 修改时间消耗记录map必须获取的锁
  static std::mutex mutex_;
  // 各类型层的时间消耗记录map
  static PtrLayerTimeStatesCollector time_states_collector_;
};

// 记录一个层的执行时间
class LayerTimeLogging {
 public:
  /**
   * 记录一个层的开始执行时间
   */
  explicit LayerTimeLogging(std::string layer_name, std::string layer_type);

  /**
   * 记录一个层的结束执行时间
   */
  ~LayerTimeLogging();

  /**
   * 输出所有类型层的运行执行时间
   */
  static void SummaryLogging();

 private:
  // 层的名称
  std::string layer_name_;
  // 层的类型
  std::string layer_type_;
  // 层的开始执行时间
  std::chrono::steady_clock::time_point start_time_;
};
}  // namespace utils
}  // namespace fantasy_infer
#endif  // FANTASY_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
