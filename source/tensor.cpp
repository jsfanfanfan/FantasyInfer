#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace fantasy_infer
{
    // 创建一个三维张量，还没有赋值
    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
    {
        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    // 创建一个一维张量
    Tensor<float>::Tensor(uint32_t size)
    {
        data_ = arma::fcube(1, size, 1);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    // 创建一个二维张量
    Tensor<float>::Tensor(uint32_t rows, uint32_t cols)
    {
        data_ = arma::fcube(rows, cols, 1);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    // 用 vector 存储形状创建张量
    Tensor<float>::Tensor(const std::vector<uint32_t> &shapes)
    {
        // 目前只支持一维、二维、三维张量
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        // shapes_ = {1, 1, 1}
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);

        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    // 拷贝构造函数: 创建新对象时，深拷贝现有对象，分配新资源并拷贝
    Tensor<float>::Tensor(const Tensor &tensor)
    {
        if (this != &tensor)
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    // 移动构造函数：转移临时对象资源, 直接接管资源
    Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept
    {
        if (this != &tensor)
        {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    // 拷贝赋值运算符：给已存在对象赋值,深拷贝并替换内容,释放旧资源，分配新资源
    Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept
    {
        if (this != &tensor)
        {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    // 移动赋值运算符：给已存在对象赋值,转移资源并替换内容,释放旧资源，接管新资源
    Tensor<float> &Tensor<float>::operator=(const Tensor &tensor)
    {
        if (this != &tensor)
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
        return *this;
    }

    // 返回行数
    uint32_t Tensor<float>::rows() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    // 返回列数
    uint32_t Tensor<float>::cols() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    // 返回通道数
    uint32_t Tensor<float>::channels() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    // 返回元素个数
    uint32_t Tensor<float>::size() const
    {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    // 用 fcube 给 tensor 设置值，传递引用避免不必要的拷贝
    void Tensor<float>::set_data(const arma::fcube &data)
    {
        CHECK(data.n_rows == this->data_.n_rows)
            << data.n_rows << " != " << this->data_.n_rows;
        CHECK(data.n_cols == this->data_.n_cols)
            << data.n_cols << " != " << this->data_.n_cols;
        CHECK(data.n_slices == this->data_.n_slices)
            << data.n_slices << " != " << this->data_.n_slices;
        this->data_ = data;
    }

    // const 表示这个成员函数不会修改对象的任何成员变量
    bool Tensor<float>::empty() const { return this->data_.empty(); }

    // 返回值，不可修改值
    float Tensor<float>::index(uint32_t offset) const
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    // 返回引用，可以修改
    float &Tensor<float>::index(uint32_t offset)
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    std::vector<uint32_t> Tensor<float>::shapes() const
    {
        CHECK(!this->data_.empty());
        return {this->channels(), this->rows(), this->cols()};
    }

    // 可写
    arma::fcube &Tensor<float>::data() { return this->data_; }

    // 只读
    const arma::fcube &Tensor<float>::data() const { return this->data_; }

    // 返回一个 slice（channel） 数据，可写
    arma::fmat &Tensor<float>::slice(uint32_t channel)
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    // 只读
    const arma::fmat &Tensor<float>::slice(uint32_t channel) const
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    // 返回特定位置的元素，不可修改
    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }
    
    // 返回特定位置的元素，可修改
    float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col)
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    // tensor 四周填充
    /***
     * @param pads 填充的维度，顺序为上、下、左、右
     * @param padding_value 填充的数值
     */
    void Tensor<float>::Padding(const std::vector<uint32_t> &pads,
                                float padding_value)
    {
        CHECK(!this->data_.empty());
        CHECK_EQ(pads.size(), 4);

        // 四周填充的维度
        uint32_t pad_rows1 = pads.at(0); // up
        uint32_t pad_rows2 = pads.at(1); // bottom
        uint32_t pad_cols1 = pads.at(2); // left
        uint32_t pad_cols2 = pads.at(3); // right

        // 获取原始维度
        uint32_t orig_rows = this->rows();
        uint32_t orig_cols = this->cols();
        uint32_t channels = this->channels();

        // 计算填充后的新维度
        uint32_t new_rows = orig_rows + pad_rows1 + pad_rows2;
        uint32_t new_cols = orig_cols + pad_cols1 + pad_cols2;

        // 创建新的填充后的 cube 并初始化
        arma::fcube padded_cube(new_rows, new_cols, channels);
        padded_cube.fill(padding_value);

        // 将原始数据复制到填充后的正确位置
        for (uint32_t c = 0; c < channels; ++c)
        {
            // 获取当前通道的原始数据和填充后的数据
            const arma::fmat &orig_slice = this->data_.slice(c);
            arma::fmat &padded_slice = padded_cube.slice(c);

            // 将原始数据复制到填充矩阵的中心位置
            // 起始位置: (pad_rows1, pad_cols1)
            // 结束位置: (pad_rows1 + orig_rows - 1, pad_cols1 + orig_cols - 1)
            padded_slice.submat(pad_rows1, pad_cols1, pad_rows1 + orig_rows - 1,
                                pad_cols1 + orig_cols - 1) = orig_slice;
        }

        // 更新数据
        this->data_ = std::move(padded_cube);

        // 更新形状信息
        if (channels == 1 && new_rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{new_cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{new_rows, new_cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, new_rows, new_cols};
        }
    }

    // tensor 单一值赋值
    void Tensor<float>::Fill(float value)
    {
        CHECK(!this->data_.empty());
        this->data_.fill(value);
    }

    // 用 vector 给 tensor 填充值，支持行主序和列主序两种方式
    void Tensor<float>::Fill(const std::vector<float> &values, bool row_major)
    {
        CHECK(!this->data_.empty());
        const uint32_t total_elems = this->data_.size();
        CHECK_EQ(values.size(), total_elems);
        if (row_major)
        {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t planes = rows * cols;
            const uint32_t channels = this->data_.n_slices;

            for (uint32_t i = 0; i < channels; ++i)
            {
                auto &channel_data = this->data_.slice(i);
                const arma::fmat &channel_data_t =
                    arma::fmat(values.data() + i * planes, this->cols(), this->rows());
                channel_data = channel_data_t.t();
            }
        }
        else
        {
            std::copy(values.begin(), values.end(), this->data_.memptr());
        }
    }

    void Tensor<float>::Show()
    {
        for (uint32_t i = 0; i < this->channels(); ++i)
        {
            LOG(INFO) << "Channel: " << i;
            LOG(INFO) << "\n"
                      << this->data_.slice(i);
        }
    }

    // 对 tensor 进行 flatten 操作，变成一维向量
    void Tensor<float>::Flatten(bool row_major)
    {
        CHECK(!this->data_.empty());
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t planes = rows * cols;
        const uint32_t channels = this->data_.n_slices;
        const uint32_t total_elems = this->data_.size();
        std::vector<uint32_t> shapes = {total_elems};
        this->Reshape(shapes, row_major);
    }

    // 随机数赋值
    void Tensor<float>::Rand()
    {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    // 赋值全 1
    void Tensor<float>::Ones()
    {
        CHECK(!this->data_.empty());
        this->Fill(1.f);
    }

    // 对 tensor 中的每个元素进行变换，使用传入的 filter 函数
    void Tensor<float>::Transform(const std::function<float(float)> &filter)
    {
        CHECK(!this->data_.empty());
        this->data_.transform(filter);
    }
    
    // 返回张量的原始形状信息，维度顺序为 channels, rows, cols
    const std::vector<uint32_t> &Tensor<float>::raw_shapes() const
    {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 3);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    // 对于 tensor 进行 reshape 操作，改变其形状但不改变数据的顺序，支持行主序和列主序两种方式
    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes,
                                bool row_major)
    {
        CHECK(!this->data_.empty());
        CHECK(!shapes.empty());
        const uint32_t origin_size = this->size();
        const uint32_t current_size =
            std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        std::vector<float> values;
        if (row_major)
        {
            values = this->values(true);
        }
        if (shapes.size() == 3)
        {
            this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
        }
        else if (shapes.size() == 2)
        {
            this->data_.reshape(shapes.at(0), shapes.at(1), 1);
            this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
        }
        else
        {
            this->data_.reshape(1, shapes.at(0), 1);
            this->raw_shapes_ = {shapes.at(0)};
        }

        if (row_major)
        {
            this->Fill(values, true);
        }
    }

    // 返回张量的原始数据指针
    float *Tensor<float>::raw_ptr()
    {
        CHECK(!this->data_.empty());
        return this->data_.memptr();
    }

    // 返回张量的原始数据指针，带偏移量
    float *Tensor<float>::raw_ptr(uint32_t offset)
    {
        const uint32_t size = this->size();
        CHECK(!this->data_.empty());
        CHECK_LT(offset, size);
        return this->data_.memptr() + offset;
    }

    // 返回一个 vector，包含张量中的所有数据，支持行主序和列主序两种方式
    std::vector<float> Tensor<float>::values(bool row_major)
    {
        CHECK_EQ(this->data_.empty(), false);
        std::vector<float> values(this->data_.size());

        if (!row_major)
        {
            std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
                      values.begin());
        }
        else
        {
            uint32_t index = 0;
            for (uint32_t c = 0; c < this->data_.n_slices; ++c)
            {
                const arma::fmat &channel = this->data_.slice(c).t();
                std::copy(channel.begin(), channel.end(), values.begin() + index);
                index += channel.size();
            }
            CHECK_EQ(index, values.size());
        }
        return values;
    }

    // 返回第 index 个矩阵的原始数据指针
    float *Tensor<float>::matrix_raw_ptr(uint32_t index)
    {
        CHECK_LT(index, this->channels());
        uint32_t offset = index * this->rows() * this->cols();
        CHECK_LE(offset, this->size());
        float *mem_ptr = this->raw_ptr() + offset;
        return mem_ptr;
    }
} // namespace fantasy_infer
