#include "dvs_tracking/lk_se3.hpp"

#include <glog/logging.h>

#include <boost/range/irange.hpp>
#include <random>

#include "dvs_tracking/weight_functions.hpp"
#include "evo_utils/interpolation.hpp"

#define DEBUG_PRINT_LIMITS

//#define USE_WEIGHTED_UPDATE
//#define BLUR_EVENT_IMAGE

void LKSE3::projectMap() {
    static std::vector<float> z_values;
    z_values.clear();
    // 当前帧可见的子地图
    map_local_->clear();
    // 参考帧的相机分辨率
    cv::Size s = c_ref_.fullResolution();
    // 参考帧的深度图
    cv::Mat &depthmap = depth_ref_;
    depthmap = cv::Mat(s, CV_32F, cv::Scalar(0.));
    cv::Mat img(s, CV_32F, cv::Scalar(0.));
    
    // Number of keypoints of the reference frame visible in
    // the current frame
    n_visible_ = 0;
    size_t n_points = 0;
    
    Eigen::Affine3f T_ref_world = (T_world_kf_ * T_kf_ref_).inverse();
    //  map_ Map (built in the reference keyframe c_ref_)
    //  投影在像平面上即可 
    //   *** 参考帧应该指的是当前位置的边地图
    for (const auto &P : map_->points) {
        // 在ref处建立的世界地图点
        Eigen::Vector3f p(P.x, P.y, P.z);
        p = T_ref_world * p;
        // 使用参考帧的内参投影
        p[0] = p[0] / p[2] * c_ref_.fx() + c_ref_.cx();
        p[1] = p[1] / p[2] * c_ref_.fy() + c_ref_.cy();
        z_values.push_back(p[2]);
        // 计算的是总地图点数目
        ++n_points;
        // 确保在像平面上
        if (p[0] < 0 || p[1] < 0) continue;
        int x = p[0] + .5f, y = p[1] + .5f;
        if (x >= s.width || y >= s.height) continue;
        float z = p[2];
        // 参考帧的深度图的修正，相同位置仅保留最小的
        // Minimalistic occlusion detection
        float &depth = depthmap.at<float>(y, x);
        if (depth == 0.)
            depth = z;
        else if (depth > z)
            depth = z;
        // img用来标识 map_local_用来记录 n_visible_用来记录
        // 在参考帧上可见的地图点
        img.at<float>(y, x) = 1.;
        map_local_->push_back(P);
        ++n_visible_;
    }
    
    const int k = map_blur_;
    cv::GaussianBlur(img, img, cv::Size(k, k), 0.);
    cv::GaussianBlur(depthmap, depthmap, cv::Size(k, k), 0.);
    // 除法： depthmap只留下 local map 中的 地图点深度
    depthmap /= img;
    
    ref_img_ = img;

    // Compute Median Depth 
    // 使用全部地图点，仅排序第 z_values.begin() + z_values.size() / 2 
    std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2,
                     z_values.end());
    depth_median_ = z_values[z_values.size() / 2];
    // 可见率
    kf_visibility_ = static_cast<float>(n_visible_) / n_points;

    precomputeReferenceFrame();
}

void LKSE3::precomputeReferenceFrame() {
    cv::Mat grad_x_img, grad_y_img;
    cv::Sobel(ref_img_, grad_x_img, CV_32F, 1, 0);
    cv::Sobel(ref_img_, grad_y_img, CV_32F, 0, 1);

    // Image Pointers
    float *depth_ref = depth_ref_.ptr<float>(0),
          *grad_x = grad_x_img.ptr<float>(0),
          *grad_y = grad_y_img.ptr<float>(0);

    int w = ref_img_.cols, h = ref_img_.rows;
    
    // built Keypoints in reference frame
    keypoints_.clear();
    Vector8 vec = Vector8::Zero();
    // 共享地址 vec8 -> 映射到 vec6
    Eigen::Map<const Vector6> vec6(&vec(0));

    for (size_t y = 0; y != h; ++y) {
        // 
        float v = ((float)y - c_ref_.cy()) / c_ref_.fy();
        for (size_t x = 0; x != w; ++x) {
            size_t offset = y * w + x;       
            float z = depth_ref[offset];
     
            float pixel_value = ref_img_.at<float>(y, x);
            // 参考帧上可见的点 都是关键点，需要计算导数
            if (pixel_value < .01) continue;
            // 得到的 u v  u*z v*z 即 恢复
            float u = ((float)x - c_ref_.cx()) / c_ref_.fx();
            // 导数值
            float gx = grad_x[offset] * c_ref_.fx(),
                  gy = grad_y[offset] * c_ref_.fy();
            Vector8 v1, v2;
            // 最后两行本来就是 0
            v1 << -1. / z, 0., u / z, u * v, -(1. + u * u), v, 0., 0.;
            v2 << 0., -1. / z, v / z, 1 + v * v, -u * v, -u, 0., 0.;
            vec = gx * v1 + gy * v2;
            // pixel_value
            keypoints_.push_back(Keypoint(Eigen::Vector3f(u * z, v * z, z),
                                          pixel_value, vec,
                                          vec6 * vec6.transpose()));
        }
    }

    // Stochastic Gradient Descent
    static std::seed_seq seq{1, 2, 3, 4, 5};
    static std::mt19937 g(seq);
    std::shuffle(keypoints_.begin(), keypoints_.end(), g);
    // 
    batches_ = std::ceil(keypoints_.size() / batch_size_);
}

// 计算dx
void LKSE3::updateTransformation(const int offset, const int N,
                                 size_t pyr_lvl) {
    
    static Eigen::MatrixXf H;
    static Eigen::VectorXf Jres, dx;
    // 取出图像金字塔的一个 并生成指针
    const cv::Mat &img = pyr_new_[pyr_lvl];
    const float *new_img = img.ptr<float>(0);
    float scale = std::pow(2.f, (float)pyr_lvl);
    // 根据图像金字塔缩放
    float fx = fx_ / scale, fy = fy_ / scale, cx = cx_ / scale,
          cy = cy_ / scale;
    
    size_t w = img.cols, h = img.rows;
    H = Matrix6::Zero();
    Jres = Vector8::Zero();
    for (auto i = offset; i != offset + N; ++i) {
        const Keypoint &k = keypoints_[i];
        // 用更新后的T_cur_ref_
        Eigen::Vector3f p = T_cur_ref_ * k.P;
        float u = p[0] / p[2] * fx + cx, v = p[1] / p[2] * fy + cy;
        // const float *new_img = img.ptr<float>(0);
        // 因为关键点投影到当前帧为浮点数，所以双线性插值得其结果
        float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v);
        // 排除关键点失效 和 残差 过大 的情况
        // k.pixel_value 为 参考帧上的值
        if (I_new == -1.f) continue;
        float res = I_new - k.pixel_value;
        if (res >= .95f) continue;
        // noalias() 是为了 +=
        Jres.noalias() += k.J * res;
        // 6 * 6
        H.noalias() += k.JJt;
    }
    // scale 是 图像金字塔的缩放
    dx = H.ldlt().solve(Jres.head<6>() * scale);
    // 错误 return
    if ((bool)std::isnan((float)dx[0])) {
        LOG(WARNING) << "Matrix close to singular!";
        return;
    }
#ifdef USE_WEIGHTED_UPDATE
    // 根据残差加权
    // t R 使用不同的 加权 
    // weight_scale_trans_  weight_scale_rot_
    for (size_t i = 0; i != 3; ++i)
        dx[i] *= weight_functions::Tukey(dx[i] * weight_scale_trans_ /
                                         depth_median_);
    for (size_t i = 3; i != 6; ++i)
        dx[i] *= weight_functions::Tukey(dx[i] * weight_scale_rot_);
#endif
    // ref -> cur 
    // T_cur_ref_是估计值 保存着上一次的结果
    // x_是当前这次的结果
    T_cur_ref_ *= SE3::exp(dx).matrix();
    x_ += dx;
}

void LKSE3::trackFrame() {
    // T_ref_cam_  cur->ref
    T_cur_ref_ = T_ref_cam_.inverse();
    x_.setZero();
    // 从最小图像开始，每次选一个batch_size_大小，(iter % batches_) = 0、1、2
    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl) {
        for (size_t iter = 0; iter != max_iterations_; ++iter) {
            updateTransformation((iter % batches_) * batch_size_, batch_size_,
                                 lvl - 1);
        }
    }
}

// 在当前帧上画
void LKSE3::drawEvents(EventQueue::iterator ev_first,
                       EventQueue::iterator ev_last, cv::Mat &out) {
    // Precompute rectification table
    static std::vector<cv::Point> points;
    static std::vector<Eigen::Vector4f> weights;
    if (points.size() == 0) {   
        cv::Rect rect(0, 0, width_ - 1, height_ - 1);
        for (size_t y = 0; y != height_; ++y)
            for (size_t x = 0; x != width_; ++x) {
                // c_
                cv::Point2d p = c_.rectifyPoint(cv::Point(x, y));
                cv::Point tl(std::floor(p.x), std::floor(p.y));
                Eigen::Vector4f w(0, 0, 0, 0);
                if (rect.contains(tl)) {
                    const float fx = p.x - tl.x, fy = p.y - tl.y;
                    w[0] = (1.f - fx) * (1.f - fy);
                    w[1] = (fx) * (1.f - fy);
                    w[2] = (1.f - fx) * (fy);
                    w[3] = (fx) * (fy);
                } else {
                    tl.x = -1;
                }
                points.push_back(tl);
                weights.push_back(w);
            }
    }
    // Draw Events
    auto draw = [](float &p, const float val) { p = std::min(p + val, 1.f); };
    out = cv::Scalar(0);
    for (auto e = ev_first; e != ev_last; ++e) {
        const cv::Point &p = points[e->x + e->y * width_];
        if (p.x == -1) continue;
        const Eigen::Vector4f &w = weights[e->x + e->y * width_];
        draw(out.at<float>(p.y, p.x), w[0]);
        draw(out.at<float>(p.y, p.x + 1), w[1]);
        draw(out.at<float>(p.y + 1, p.x), w[2]);
        draw(out.at<float>(p.y + 1, p.x + 1), w[3]);
    }
}

//最近邻 画 不累加
void LKSE3::drawEventsNN(EventQueue::iterator ev_first,
                         EventQueue::iterator ev_last, cv::Mat &out) {
    // Precompute rectification table
    static std::vector<cv::Point> points;
    if (points.size() == 0) {
        for (size_t y = 0; y != height_; ++y)
            for (size_t x = 0; x != width_; ++x) {
                cv::Point p = c_.rectifyPoint(cv::Point(x, y));
                if (!rect_.contains(p)) p.x = -1;
                points.push_back(p);
            }
    }
    out = cv::Scalar(0);
    for (auto e = ev_first; e != ev_last; ++e) {
        const cv::Point &p = points[e->x + e->y * width_];
        if (p.x == -1) continue;
        out.at<float>(p) = 1.;
    }
#ifdef BLUR_EVENT_IMAGE
    const int k = map_blur_;
    cv::GaussianBlur(out, out, cv::Size(k, k), 0.);
#endif
}
