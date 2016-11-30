#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/hdfs_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
        HdfsDataLayer<Dtype>::~HdfsDataLayer<Dtype>() {
        }

    template <typename Dtype>
        void HdfsDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            const std::string source = this->layer_param_.hdfs_data_param().source();
            Status status = hdfs.NewRandomAccessFile(source, &raf);
            CHECK_EQ(status.ok(), true) << "can't open file: " << status;

            data_transformer_.reset(new DataTransformer<Dtype>(transform_param_, this->phase_));
            data_transformer_->InitRand();

            const int size_per_datum = this->layer_param_.hdfs_data_param().size_per_datum();
            const int batch_size = this->layer_param_.hdfs_data_param().batch_size();
            CHECK_GT(batch_size, 0) << "Positive batch size required";

            char* ch = new char[size_per_datum + 1]();
            StringPiece sp;
            status = raf->Read(0, size_per_datum, &sp, ch);
            CHECK_EQ(status.ok(), true) << "can't read file: " << status;

            Datum datum;
            bool ret = datum.ParseFromString(sp.ToString());
            CHECK_EQ(ret, true) << "proto can't parse from string";

            // Use data_transformer to infer the expected blob shape from a Datum.
            vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

            top_shape[0] = batch_size;
            top[0]->Reshape(top_shape);
            LOG(INFO) << "output data size: " << top[0]->num() << ","
                << top[0]->channels() << "," << top[0]->height() << ","
                << top[0]->width();

            // label
            vector<int> label_shape(1, batch_size);
            top[1]->Reshape(label_shape);

            delete ch;
        }

    template <typename Dtype>
        void HdfsDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {

        }

    template <typename Dtype>
        void HdfsDataLayer<Dtype>::Forward_cpu(const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
            const int size_per_datum = this->layer_param_.hdfs_data_param().size_per_datum();
            const int batch_size = this->layer_param_.hdfs_data_param().batch_size();

            unsigned long long size = 0;
            const std::string source = this->layer_param_.hdfs_data_param().source();
            Status status = hdfs.GetFileSize(source, &size);
            CHECK_EQ(status.ok(), true) << "can't get size of file: " << status;

            Dtype* top_label = top[1]->mutable_cpu_data();

            std::vector<Datum> datums;
            char* ch = new char[size_per_datum + 1]();
            for (int i = 0; i < batch_size; ++i) {
                StringPiece sp;
                status = raf->Read(offset, size_per_datum, &sp, ch);

                Datum datum;
                datum.ParseFromString(sp.ToString());

                datums.push_back(datum);
                offset += size_per_datum;
                if (offset == size) {
                    offset = 0;
                }

                top_label[i] = datum.label();
            }

            delete ch;
            this->data_transformer_->Transform(datums, top[0]);
        }

    INSTANTIATE_CLASS(HdfsDataLayer);
    REGISTER_LAYER_CLASS(HdfsData);

}  // namespace caffe
#endif  // USE_OPENCV
