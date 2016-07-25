#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    using std::string;
    using std::vector;
    using caffe::CaffeMobile;

    int getTimeSec() {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (int)now.tv_sec;
    }

    string jstring2string(JNIEnv *env, jstring jstr) {
        const char *cstr = env->GetStringUTFChars(jstr, 0);
        string str(cstr);
        env->ReleaseStringUTFChars(jstr, cstr);
        return str;
    }

    JNIEXPORT void JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setNumThreads(
                JNIEnv *env, jobject thiz, jint numThreads) {
            int num_threads = numThreads;
#ifdef USE_EIGEN
            omp_set_num_threads(num_threads);
#else
            openblas_set_num_threads(num_threads);
#endif
        }

    JNIEXPORT jlong JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_create(
            JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath) {
        return (jlong)new CaffeMobile(jstring2string(env, modelPath),
                jstring2string(env, weightsPath));
    }

    JNIEXPORT void JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanFile(
                JNIEnv *env, jlong thiz, jstring meanFile) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            caffe_mobile->SetMean(jstring2string(env, meanFile));
        }

    JNIEXPORT void JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanValues(
                JNIEnv *env, jlong thiz, jfloatArray meanValues) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            int num_channels = env->GetArrayLength(meanValues);
            jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
            vector<float> mean_values(ptr, ptr + num_channels);
            caffe_mobile->SetMean(mean_values);
        }

    JNIEXPORT void JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setScale(JNIEnv *env,
                jlong thiz, jfloat scale) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            caffe_mobile->SetScale(scale);
        }

    JNIEXPORT jintArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictImage(JNIEnv *env,
                jlong thiz, jstring imgPath, jint k) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            vector<int> top_k =
                caffe_mobile->PredictTopK(jstring2string(env, imgPath), k);

            jintArray result;
            result = env->NewIntArray(k);
            if (result == NULL) {
                return NULL; /* out of memory error thrown */
            }
            // move from the temp structure to the java structure
            env->SetIntArrayRegion(result, 0, k, &top_k[0]);
            return result;
        }


    JNIEXPORT jobjectArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictJPEG(JNIEnv* env, jlong thiz, jint front1orback0, jint orientCase, jbyteArray jpegdata, jint k, jstring jclsname) {
            LOG(INFO) << "native predictJPEG() called.";

            int len = env->GetArrayLength(jpegdata);
            LOG(INFO) << "Length: " << len;

            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;

            jbyte* jpegjData = env->GetByteArrayElements(jpegdata, 0);
            caffe::CaffeMobile::predictWithScore prediction = caffe_mobile->PredictJPEGTopK((int) front1orback0, (int) orientCase, (unsigned char*)jpegjData, len, (int) k);

            env->ReleaseByteArrayElements(jpegdata, jpegjData, JNI_ABORT);

            const char *clsname = env->GetStringUTFChars(jclsname, 0);
            jclass clsPredictScore = env->FindClass(clsname);
            env->ReleaseStringUTFChars(jclsname, clsname);

            if(clsPredictScore != NULL) LOG(INFO) << "successfully created class";
            jmethodID mtdPredictScore = env->GetMethodID(clsPredictScore, "<init>", "(IF)V");
            if(mtdPredictScore != NULL) LOG(INFO) << "successfully created constructor";
            int kpred = prediction.idx.size();
            jobjectArray pred = env->NewObjectArray(k, clsPredictScore, NULL);

            for (int i = 0; i < kpred; i++) {
                jobject predItem = env->NewObject(clsPredictScore, mtdPredictScore, prediction.idx[i], prediction.scr[i]);
                env->SetObjectArrayElement(pred, i, predItem);
            }

            return pred;
        }


    JNIEXPORT jobjectArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictFrame(JNIEnv* env, jlong thiz, jint width, jint height, jint front1orback0, jint orientCase, jbyteArray frmdata, jint k, jstring jclsname) {
            // LOG(INFO) << "native predictFrame() called.";

            // int len = env->GetArrayLength(frmdata);
            // LOG(INFO) << "Length: " << len;

            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;

            jbyte* frmjData = env->GetByteArrayElements(frmdata, 0);
            caffe::CaffeMobile::predictWithScore prediction = caffe_mobile->PredictFrameTopK((int) width, (int) height, (int) front1orback0, (int) orientCase, (unsigned char*)frmjData, (int) k);

            env->ReleaseByteArrayElements(frmdata, frmjData, JNI_ABORT);

            const char *clsname = env->GetStringUTFChars(jclsname, 0);
            jclass clsPredictScore = env->FindClass(clsname);
            env->ReleaseStringUTFChars(jclsname, clsname);

            if(clsPredictScore != NULL) LOG(INFO) << "successfully created class";
            jmethodID mtdPredictScore = env->GetMethodID(clsPredictScore, "<init>", "(IF)V");
            if(mtdPredictScore != NULL) LOG(INFO) << "successfully created constructor";
            int kpred = prediction.idx.size();
            jobjectArray pred = env->NewObjectArray(k, clsPredictScore, NULL);

            for (int i = 0; i < kpred; i++) {
                jobject predItem = env->NewObject(clsPredictScore, mtdPredictScore, prediction.idx[i], prediction.scr[i]);
                env->SetObjectArrayElement(pred, i, predItem);
            }

            return pred;
        }


    JNIEXPORT jobjectArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeatures(
                JNIEnv *env, jlong thiz, jstring imgPath, jstring blobNames) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            vector<vector<float> > features = caffe_mobile->ExtractFeatures(
                    jstring2string(env, imgPath), jstring2string(env, blobNames));

            jobjectArray array2D =
                env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
            for (size_t i = 0; i < features.size(); ++i) {
                jfloatArray array1D = env->NewFloatArray(features[i].size());
                if (array1D == NULL) {
                    return NULL; /* out of memory error thrown */
                }
                // move from the temp structure to the java structure
                env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
                env->SetObjectArrayElement(array2D, i, array1D);
            }
            return array2D;
        }

    JNIEXPORT jobjectArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeaturesCVMat(JNIEnv* env, jlong thiz, jlong matAddr, jstring blobNames) {

            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            const cv::Mat &cvMat = *(cv::Mat *) matAddr;

            vector<vector<float> > features = caffe_mobile->ExtractFeaturesCVMat(cvMat, jstring2string(env, blobNames));

            jobjectArray array2D =
                env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
            for (size_t i = 0; i < features.size(); ++i) {
                jfloatArray array1D = env->NewFloatArray(features[i].size());
                if (array1D == NULL) {
                    return NULL; /* out of memory error thrown */
                }
                // move from the temp structure to the java structure
                env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
                env->SetObjectArrayElement(array2D, i, array1D);
            }
            return array2D;
        }

    JNIEXPORT jobjectArray JNICALL
        Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeaturesCVBatch(
                JNIEnv *env, jlong thiz, jlong batchAddr, jstring blobName, jint endLayerOffset) {
            CaffeMobile *caffe_mobile = (CaffeMobile*) thiz;
            const vector<cv::Mat> &cvMatBatch = *(vector<cv::Mat>*) batchAddr;

            // for testing
            // vector<cv::Mat> test;
            // for (int i = 0; i < 10; i++) {
                // cv::Mat tmp(128, 128, CV_8UC1, cv::Scalar::all(i));
                // LOG(INFO) << "test: " << cv::mean(tmp);
                // test.push_back(tmp);
            // }
            // const vector<cv::Mat> &cvMatBatch = test;

            LOG(INFO) << "Get cvMatBatch size: " << cvMatBatch.size();
            vector<vector<float> > features = caffe_mobile->ExtractFeaturesCVBatch(cvMatBatch, jstring2string(env, blobName), (int) endLayerOffset);
            LOG(INFO) << "Features size: " << features.size();
            jobjectArray array2D =
                env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
            for (size_t i = 0; i < features.size(); ++i) {
                jfloatArray array1D = env->NewFloatArray(features[i].size());
                if (array1D == NULL) {
                    return NULL; /* out of memory error thrown */
                }
                // move from the temp structure to the java structure
                env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
                env->SetObjectArrayElement(array2D, i, array1D);
            }
            return array2D;
        }

    JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
        JNIEnv *env = NULL;
        jint result = -1;

        if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
            LOG(FATAL) << "GetEnv failed!";
            return result;
        }

        return JNI_VERSION_1_6;
    }

#ifdef __cplusplus
}
#endif
