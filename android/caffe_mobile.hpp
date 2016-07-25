#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>

using std::string;
using std::vector;

namespace caffe {

    class CaffeMobile {
        public:
            ~CaffeMobile();

            CaffeMobile(const string &model_path, const string &weights_path);

            void SetMean(const string &mean_file);

            void SetMean(const vector<float> &mean_values);

            void SetScale(const float scale);

            vector<int> PredictTopK(const string &img_path, int k);

            vector<vector<float> > ExtractFeatures(const string &img_path,
                    const string &str_blob_names);

            vector<vector<float> > ExtractFeaturesCVMat(const cv::Mat &img, const string &blob_names);

            vector<vector<float> > ExtractFeaturesCVBatch(const vector<cv::Mat> &imgs, const string &blob_name, int endLayerOffset = 0);

            struct predictWithScore {
                vector<int> idx;
                vector<float> scr;
            };

            predictWithScore PredictFrameTopK(int width, int height, int front1orback0, int orientCase, unsigned char* frmcData, int k=5);

            predictWithScore PredictJPEGTopK(int front1orback0, int orientCase, unsigned char* jpegcData, int jpegcDataLen, int k=5);

        private:

            void Preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels);
            void PreprocessBatch(const vector<cv::Mat> &imgs, vector<cv::Mat> *input_channels);

            void WrapInputLayer(std::vector<cv::Mat> *input_channels);
            void WrapInputLayerBatch(std::vector<cv::Mat> *input_channels, int batch_size);

            vector<float> Forward(const string &filename);
            vector<float> ForwardCVMat(const cv::Mat &img);
            vector<vector<float> > ForwardCVMatBatch(const vector<cv::Mat> &imgs, int endLayerOffset = 0);

            shared_ptr<Net<float > > net_;
            cv::Size input_geometry_;
            int num_channels_;
            cv::Mat mean_;
            float scale_;


    };

} // namespace caffe

#endif
