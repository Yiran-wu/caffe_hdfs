#ifndef CAFFE_HDFS_DATA_LAYER_HPP_
#define CAFFE_HDFS_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/hdfs/hadoop_file_system.h"

namespace caffe {

    /**
     * @brief Provides data to the Net from hdfs.
     *
     */
    template <typename Dtype>
        class HdfsDataLayer : public Layer<Dtype> {
            public:
                explicit HdfsDataLayer(const LayerParameter& param): Layer<Dtype>(param),
                                    transform_param_(param.transform_param()), offset(0) {}
                virtual ~HdfsDataLayer();
                virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
                virtual void Reshape(const vector<Blob<Dtype>*>& top,
                                        const vector<Blob<Dtype>*>& bottom);

                virtual inline const char* type() const { return "HdfsData"; }
                virtual inline int ExactNumBottomBlobs() const { return 0; }
                virtual inline int ExactNumTopBlobs() const { return 2; }

                virtual void Forward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<Blob<Dtype>*>& bottom);
                virtual void Forward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<Blob<Dtype>*>& bottom) {}

                virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
                virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

            protected:
                TransformationParameter transform_param_;
                shared_ptr<DataTransformer<Dtype> > data_transformer_;
                HadoopFileSystem hdfs;
                std::shared_ptr<RandomAccessFile> raf;
                int offset;
        };


}  // namespace caffe

#endif  // CAFFE_HDFS_DATA_LAYER_HPP_
