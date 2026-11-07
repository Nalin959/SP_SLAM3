#include "cuda_utils.h"
#include "save_mat.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <unistd.h>
#include <atomic>
#include <cuda_runtime.h>

#include "SuperPoint.h"
#include "LightGlue.h"

using namespace std;
using namespace cv;

// --- EKF Log Synchronizer ---
struct EKFState {
    double time;
    double pn, pe;
    double yaw;
};

class LogSynchronizer {
public:
    LogSynchronizer(const string& csv_path) {
        ifstream file(csv_path);
        if (!file.is_open()) {
            cerr << "Failed to open EKF log file: " << csv_path << endl;
            return;
        }
        string line;
        bool first_line = true;
        
        int time_idx = -1, pn_idx = -1, pe_idx = -1, yaw_idx = -1;
        double prev_yaw = 0.0;
        bool first_yaw = true;
        
        while (getline(file, line)) {
            stringstream ss(line);
            string cell;
            vector<string> row;
            while (getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            if (row.empty()) continue;
            
            if (first_line) {
                for (size_t i = 0; i < row.size(); ++i) {
                    if (row[i] == "timestamp_us") time_idx = i;
                    else if (row[i] == "pn") pn_idx = i;
                    else if (row[i] == "pe") pe_idx = i;
                    else if (row[i] == "yaw") yaw_idx = i;
                }
                first_line = false;
                continue;
            }
            
            if (time_idx >= 0 && pn_idx >= 0 && pe_idx >= 0 && yaw_idx >= 0) {
                EKFState s;
                static double first_timestamp = -1;
                double ts = stod(row[time_idx]);
                if (first_timestamp < 0) first_timestamp = ts;
                s.time = (ts - first_timestamp) / 1e6;
                
                s.pn = stod(row[pn_idx]);
                s.pe = stod(row[pe_idx]);
                
                double raw_yaw = stod(row[yaw_idx]);
                // Unwrap yaw
                if (first_yaw) {
                    s.yaw = raw_yaw;
                    first_yaw = false;
                } else {
                    double diff = raw_yaw - prev_yaw;
                    while (diff > 180.0) diff -= 360.0;
                    while (diff < -180.0) diff += 360.0;
                    s.yaw = prev_yaw + diff;
                }
                prev_yaw = s.yaw;
                states.push_back(s);
            }
        }
        cout << "Loaded " << states.size() << " EKF states." << endl;
    }
    
    EKFState getState(int frame_idx, double fps = 59.94) {
        double t = frame_idx / fps;
        auto it = lower_bound(states.begin(), states.end(), t, [](const EKFState& s, double val) {
            return s.time < val;
        });
        
        if (it == states.end()) return states.back();
        if (it == states.begin()) return states.front();
        
        auto it_prev = it - 1;
        double dt = it->time - it_prev->time;
        if (dt < 1e-6) return *it;
        
        double alpha = (t - it_prev->time) / dt;
        EKFState res;
        res.time = t;
        res.pn = it_prev->pn + alpha * (it->pn - it_prev->pn);
        res.pe = it_prev->pe + alpha * (it->pe - it_prev->pe);
        res.yaw = it_prev->yaw + alpha * (it->yaw - it_prev->yaw);
        return res;
    }

private:
    vector<EKFState> states;
};

// --- Simple DBSCAN for 2D points ---
void simpleDBSCAN(const vector<Point2f>& pts, float eps, int minPts, vector<int>& labels, vector<Point2f>& cluster_centers) {
    int n = pts.size();
    labels.assign(n, -1);
    int cluster_id = 0;
    
    for (int i = 0; i < n; i++) {
        if (labels[i] != -1) continue;
        
        vector<int> neighbors;
        for (int j = 0; j < n; j++) {
            float dx = pts[i].x - pts[j].x;
            float dy = pts[i].y - pts[j].y;
            if (dx*dx + dy*dy <= eps*eps) neighbors.push_back(j);
        }
        
        if (neighbors.size() < minPts) {
            labels[i] = -2; // noise
            continue;
        }
        
        labels[i] = cluster_id;
        for (size_t k = 0; k < neighbors.size(); k++) {
            int p = neighbors[k];
            if (labels[p] == -2) labels[p] = cluster_id;
            if (labels[p] != -1) continue;
            
            labels[p] = cluster_id;
            vector<int> p_neighbors;
            for (int j = 0; j < n; j++) {
                float dx = pts[p].x - pts[j].x;
                float dy = pts[p].y - pts[j].y;
                if (dx*dx + dy*dy <= eps*eps) p_neighbors.push_back(j);
            }
            if (p_neighbors.size() >= minPts) {
                neighbors.insert(neighbors.end(), p_neighbors.begin(), p_neighbors.end());
            }
        }
        cluster_id++;
    }
    
    cluster_centers.resize(cluster_id);
    vector<int> cluster_counts(cluster_id, 0);
    for (int i = 0; i < cluster_id; i++) cluster_centers[i] = Point2f(0, 0);
    
    for (int i = 0; i < n; i++) {
        int id = labels[i];
        if (id >= 0) {
            cluster_centers[id] += pts[i];
            cluster_counts[id]++;
        }
    }
    
    for (int i = 0; i < cluster_id; i++) {
        if (cluster_counts[i] > 0) {
            cluster_centers[i].x /= cluster_counts[i];
            cluster_centers[i].y /= cluster_counts[i];
        }
    }
}

// --- Map Data Loader ---
struct MapData {
    vector<KeyPoint> kpts;
    Mat desc;
    int width, height;
};

MapData loadMapDataBin(const string& bin_path) {
    MapData md;
    ifstream f(bin_path, ios::binary);
    if (!f.is_open()) {
        cerr << "Failed to open " << bin_path << endl;
        return md;
    }
    
    int num_kpts;
    f.read((char*)&num_kpts, sizeof(int));
    
    int shape[2];
    f.read((char*)shape, 2 * sizeof(int));
    md.height = shape[0];
    md.width = shape[1];
    
    vector<float> raw_kpts(num_kpts * 2);
    f.read((char*)raw_kpts.data(), num_kpts * 2 * sizeof(float));
    
    md.kpts.resize(num_kpts);
    for (int i = 0; i < num_kpts; i++) {
        md.kpts[i].pt.x = raw_kpts[i*2];
        md.kpts[i].pt.y = raw_kpts[i*2+1];
        md.kpts[i].response = 1.0f;
    }
    
    md.desc.create(num_kpts, 256, CV_32F);
    f.read((char*)md.desc.data, num_kpts * 256 * sizeof(float));
    
    return md;
}

// Rotate image
void rotateImage(const Mat& src, Mat& dst, double angle) {
    Point2f center((src.cols - 1) / 2.0f, (src.rows - 1) / 2.0f);
    Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    warpAffine(src, dst, rot, src.size(), INTER_LINEAR);
}

int main(int argc, char** argv) {
    cout << "Initializing Video Localizer (C++ TRT)..." << endl;
    
    // Default Paths & Configuration
    string video_path = "../GPSDenied/Videos/Digaru/CameraRecord/VID_EO_15.mp4";
    string ekf_path = "../sensor_data/VID_EO_15/ekf_state.csv";
    string map_path = "../map_data/reference_map_2km.png";
    string map_bin_path = "../map_data/reference_map_2km_features.bin";
    int start_frame = 79120;
    int lock_threshold = 70;
    string output_video_path = "../outputs/output_localizer.mp4";

    // Parse CLI overrides: [video] [ekf] [map] [map_bin] [start_frame] [threshold] [output_video]
    if (argc > 1) video_path = argv[1];
    if (argc > 2) ekf_path = argv[2];
    if (argc > 3) map_path = argv[3];
    if (argc > 4) map_bin_path = argv[4];
    if (argc > 5) start_frame = std::stoi(argv[5]);
    if (argc > 6) lock_threshold = std::stoi(argv[6]);
    if (argc > 7) output_video_path = argv[7];

    cout << "Video Source: " << video_path << endl;
    cout << "EKF Source:   " << ekf_path << endl;
    cout << "Map Source:   " << map_path << endl;
    cout << "Start Frame:  " << start_frame << " | Lock Threshold: " << lock_threshold << " matches" << endl;
    cout << "Output Video: " << output_video_path << endl;
    
    // Load TRT engines (check models/ or root)
    string sp_engine = "../models/superpoint.engine";
    if (access(sp_engine.c_str(), F_OK) != 0) sp_engine = "../superpoint.engine";
    string lg_engine = "../models/lightglue.engine";
    if (access(lg_engine.c_str(), F_OK) != 0) lg_engine = "../lightglue.engine";

    shared_ptr<ORB_SLAM3::SuperPoint> sp_drone_model = make_shared<ORB_SLAM3::SuperPoint>(sp_engine);
    ORB_SLAM3::SPDetector sp_drone_det(sp_drone_model, false);
    
    shared_ptr<ORB_SLAM3::SuperPoint> sp_map_model = make_shared<ORB_SLAM3::SuperPoint>(sp_engine);
    ORB_SLAM3::SPDetector sp_map_det(sp_map_model, false);

    shared_ptr<ORB_SLAM3::LightGlue> lg_model = make_shared<ORB_SLAM3::LightGlue>(lg_engine);
    
    // Load EKF
    LogSynchronizer log_sync(ekf_path);
    
    // Load Map Features
    MapData global_map_features = loadMapDataBin(map_bin_path);
    cout << "Loaded global map features: " << global_map_features.kpts.size() << " kpts" << endl;
    
    // Use FlannBasedMatcher for global search
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    
    // Load Reference Map
    Mat ref_map = imread(map_path, IMREAD_GRAYSCALE);
    if (ref_map.empty()) {
        cerr << "Error: Could not load reference map from " << map_path << endl;
        return -1;
    }
    int LARGE_CROP = 1400;
    int pad = LARGE_CROP / 2;
    Mat padded_map;
    copyMakeBorder(ref_map, padded_map, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
    
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file " << video_path << endl;
        return -1;
    }
    cap.set(CAP_PROP_POS_FRAMES, start_frame);
    
    Mat frame;
    bool has_locked_on = false;
    double center_x = 0, center_y = 0;
    double last_pn = 0, last_pe = 0;
    double MAP_SCALE = 0.534563;
    
    int frame_count = 0;
    int match_count = 0;
    auto start_time = chrono::high_resolution_clock::now();
    
    // Allocate GPU buffers
    uint8_t* d_map = nullptr;
    cudaMalloc(&d_map, padded_map.total() * padded_map.elemSize());
    cudaMemcpy(d_map, padded_map.data, padded_map.total() * padded_map.elemSize(), cudaMemcpyHostToDevice);
    
    uint8_t* d_frame = nullptr;
    float* d_frame_f = nullptr;
    float* d_map_f = nullptr;
    
    cudaMalloc(&d_frame, 3840 * 2160 * 3); 
    cudaMalloc(&d_frame_f, 1024 * 1024 * sizeof(float));
    uint8_t* d_frame_180 = nullptr;
    float* d_frame_f_180 = nullptr;
    cudaMalloc(&d_frame_180, 3840 * 2160 * 3); 
    cudaMalloc(&d_frame_f_180, 1024 * 1024 * sizeof(float));
    cudaMalloc(&d_map_f, 1024 * 1024 * sizeof(float));
    
    uchar3* d_out_render = nullptr;
    cudaMalloc(&d_out_render, 2048 * 1024 * 3);
    float2* d_pts1 = nullptr;
    float2* d_pts2 = nullptr;
    cudaMalloc(&d_pts1, 4096 * sizeof(float2));
    cudaMalloc(&d_pts2, 4096 * sizeof(float2));

    // Render thread setup
    std::mutex render_mutex;
    std::queue<Mat> render_queue;
    std::atomic<bool> is_running(true);
    std::thread render_thread([&]() {
        VideoWriter out(output_video_path, VideoWriter::fourcc('m', 'p', '4', 'v'), 30, Size(2048, 1024));
        while (is_running || !render_queue.empty()) {
            Mat frame_to_write;
            {
                std::lock_guard<std::mutex> lock(render_mutex);
                if (!render_queue.empty()) {
                    frame_to_write = render_queue.front();
                    render_queue.pop();
                }
            }
            if (!frame_to_write.empty()) {
                out.write(frame_to_write);
            } else {
                std::this_thread::sleep_for(chrono::milliseconds(1));
            }
        }
        out.release();
    });

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (cap.read(frame)) { 

        int current_frame_idx = start_frame + frame_count;
        EKFState state = log_sync.getState(current_frame_idx, 29.97);
        auto t_frame_start = chrono::high_resolution_clock::now();
        if (frame.empty()) {
            break;
        }
        auto t_cap_end = chrono::high_resolution_clock::now();

        cudaMemcpyAsync(d_frame, frame.data, frame.rows * frame.step, cudaMemcpyHostToDevice, stream);
        cuda_bgr2gray_resize_normalize(d_frame, frame.cols, frame.rows, frame.step, d_frame_f, 1024, 1024, stream);
        cudaStreamSynchronize(stream);
        auto t_pre_end = chrono::high_resolution_clock::now();
        
        // --- Initialization Mode ---
        if (!has_locked_on) {
            Mat frame_180;
            cv::rotate(frame, frame_180, cv::ROTATE_180);
            cudaMemcpyAsync(d_frame_180, frame_180.data, frame.rows * frame.step, cudaMemcpyHostToDevice, stream);
            cuda_bgr2gray_resize_normalize(d_frame_180, frame.cols, frame.rows, frame.step, d_frame_f_180, 640, 360, stream);
            cudaStreamSynchronize(stream);
            
            // Need drone detect for initialization
            sp_drone_det.detectCUDA(d_frame_f_180, 360, 640);
cudaStreamSynchronize(stream);
            
            // Now block and get results
            vector<KeyPoint> drone_kpts_180;
            sp_drone_det.getKeyPoints(0.005, 0, 640, 0, 360, drone_kpts_180, true);
            if (drone_kpts_180.size() > 4096) {
                sort(drone_kpts_180.begin(), drone_kpts_180.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });
                drone_kpts_180.resize(4096);
            }
            Mat drone_desc_180;
            sp_drone_det.computeDescriptors(drone_kpts_180, drone_desc_180);

            vector<vector<DMatch>> knn_matches;
            matcher->knnMatch(drone_desc_180, global_map_features.desc, knn_matches, 2);
            
            vector<Point2f> matched_map_pts;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.85f * knn_matches[i][1].distance) {
                    matched_map_pts.push_back(global_map_features.kpts[knn_matches[i][0].trainIdx].pt);
                }
            }
            
            if (matched_map_pts.size() >= 10) {
                vector<int> labels;
                vector<Point2f> centers;
                simpleDBSCAN(matched_map_pts, 200.0f, 4, labels, centers);
                if (centers.empty() && !matched_map_pts.empty()) {
                    std::vector<float> xs, ys;
                    for (auto& p : matched_map_pts) { xs.push_back(p.x); ys.push_back(p.y); }
                    std::sort(xs.begin(), xs.end());
                    std::sort(ys.begin(), ys.end());
                    centers.push_back(Point2f(xs[xs.size()/2], ys[ys.size()/2]));
                }
                int best_matches = 0;
                // We need dense features from 1024x1024 unrotated image for LightGlue

                sp_drone_det.detectCUDA(d_frame_f, 1024, 1024);
cudaStreamSynchronize(stream);

                vector<KeyPoint> drone_kpts;

                sp_drone_det.getKeyPoints(0.005, 0, 1024, 0, 1024, drone_kpts, true);

                if (drone_kpts.size() > 4096) {

                    sort(drone_kpts.begin(), drone_kpts.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });

                    drone_kpts.resize(4096);

                }

                Mat drone_desc;

                sp_drone_det.computeDescriptors(drone_kpts, drone_desc);


                std::cout << "Evaluated " << centers.size() << " candidates:" << std::endl;
                for (size_t c = 0; c < min(centers.size(), size_t(3)); c++) {
                    std::cout << "  Candidate " << c << ": " << centers[c].x << ", " << centers[c].y << std::endl;
                    cuda_crop_map_normalize(d_map, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, centers[c].x + pad, centers[c].y + pad, -state.yaw * M_PI / 180.0f, stream);
                    cudaStreamSynchronize(stream);
                    sp_map_det.detectCUDA(d_map_f, 1024, 1024);
cudaStreamSynchronize(stream);
                    vector<KeyPoint> map_kpts;
                    sp_map_det.getKeyPoints(0.005, 0, 1024, 0, 1024, map_kpts, true);
                    if (map_kpts.size() > 4096) {
                        sort(map_kpts.begin(), map_kpts.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });
                        map_kpts.resize(4096);
                    }
                    Mat map_desc;
                    sp_map_det.computeDescriptors(map_kpts, map_desc);
                    auto lg_matches = lg_model->match(map_kpts, map_desc, drone_kpts, drone_desc, Size(1024, 1024));
                    int good_matches = lg_matches.size();
                    if (good_matches > best_matches) {
                        best_matches = good_matches;
                        center_x = centers[c].x;
                        center_y = centers[c].y;
                        match_count = good_matches;
                    }
                }
                if (best_matches >= lock_threshold) {
                    has_locked_on = true;
                    last_pn = state.pn;
                    last_pe = state.pe;
                    std::cout << "LOCKED ON WITH " << best_matches << " MATCHES AT " << center_x << "," << center_y << std::endl;
                }
            }
        } 
        // --- Tracking Mode ---
        else {
            double delta_pe = state.pe - last_pe;
            double delta_pn = state.pn - last_pn;
            center_x += (delta_pe / MAP_SCALE);
            center_y += (-delta_pn / MAP_SCALE);
            last_pn = state.pn;
            last_pe = state.pe;
            
            auto t_crop_start = chrono::high_resolution_clock::now();
            cuda_crop_map_normalize(d_map, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, center_x + pad, center_y + pad, -state.yaw * M_PI / 180.0f, stream);
            cudaStreamSynchronize(stream);
            auto t_crop_end = chrono::high_resolution_clock::now();
            
            auto t_det_start = chrono::high_resolution_clock::now();

            
            // Launch sequential
            sp_drone_det.detectCUDA(d_frame_f, 1024, 1024);
cudaStreamSynchronize(stream);
            sp_map_det.detectCUDA(d_map_f, 1024, 1024);
cudaStreamSynchronize(stream);
            
            vector<KeyPoint> drone_kpts;
            sp_drone_det.getKeyPoints(0.005, 0, 1024, 0, 1024, drone_kpts, true);
            if (drone_kpts.size() > 4096) {
                sort(drone_kpts.begin(), drone_kpts.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });
                drone_kpts.resize(4096);
            }
            vector<KeyPoint> map_kpts;
            sp_map_det.getKeyPoints(0.005, 0, 1024, 0, 1024, map_kpts, true);
            if (map_kpts.size() > 4096) {
                sort(map_kpts.begin(), map_kpts.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });
                map_kpts.resize(4096);
            }
            auto t_det_end = chrono::high_resolution_clock::now();

            auto t_desc_start = chrono::high_resolution_clock::now();
            Mat drone_desc;
            sp_drone_det.computeDescriptors(drone_kpts, drone_desc);
            Mat map_desc;
            sp_map_det.computeDescriptors(map_kpts, map_desc);
            auto t_desc_end = chrono::high_resolution_clock::now();

            auto t_match_start = chrono::high_resolution_clock::now();
            auto lg_matches = lg_model->match(map_kpts, map_desc, drone_kpts, drone_desc, Size(1024, 1024));
            match_count = lg_matches.size();
            auto t_match_end = chrono::high_resolution_clock::now();
            
            // --- FULL CUDA RENDERING ---
            std::vector<float2> pts1(match_count);
            std::vector<float2> pts2(match_count);
            for (int i = 0; i < match_count; i++) {
                pts1[i] = {drone_kpts[lg_matches[i].idx1].pt.x, drone_kpts[lg_matches[i].idx1].pt.y};
                pts2[i] = {map_kpts[lg_matches[i].idx0].pt.x, map_kpts[lg_matches[i].idx0].pt.y};
            }
            if (match_count > 0) {
                cudaMemcpy(d_pts1, pts1.data(), match_count * sizeof(float2), cudaMemcpyHostToDevice);
                cudaMemcpy(d_pts2, pts2.data(), match_count * sizeof(float2), cudaMemcpyHostToDevice);
            }
            
            cuda_render_visualization(d_frame_f, d_map_f, d_out_render, d_pts1, d_pts2, match_count, stream);
            cudaStreamSynchronize(stream);
            
            Mat match_img(1024, 2048, CV_8UC3);
            cudaMemcpy(match_img.data, d_out_render, 2048 * 1024 * 3, cudaMemcpyDeviceToHost);
            
            {
                std::lock_guard<std::mutex> lock(render_mutex);
                if (render_queue.size() < 10) render_queue.push(match_img);
            }
            
            if (frame_count % 10 == 0) {
                double crop_time = chrono::duration<double, std::milli>(t_crop_end - t_crop_start).count();
                double det_time = chrono::duration<double, std::milli>(t_det_end - t_det_start).count();
                double desc_time = chrono::duration<double, std::milli>(t_desc_end - t_desc_start).count();
                double match_time = chrono::duration<double, std::milli>(t_match_end - t_match_start).count();
                double cap_time = chrono::duration<double, std::milli>(t_cap_end - t_frame_start).count();
                double pre_time = chrono::duration<double, std::milli>(t_pre_end - t_cap_end).count();
                double total_loop_time = chrono::duration<double, std::milli>(t_match_end - t_frame_start).count();
                
                cout << "  [TIMING] Cap: " << cap_time << "ms | Pre: " << pre_time << "ms | Crop: " << crop_time << "ms | Detect(Ovl): " << det_time << "ms | Desc: " << desc_time << "ms | Match: " << match_time << "ms | TotalLoop: " << total_loop_time << "ms" << endl;
            }
        }
        
        frame_count++;
        if (frame_count % 10 == 0) {
            auto end_time = chrono::high_resolution_clock::now();
            double elap = chrono::duration<double>(end_time - start_time).count();
            cout << "  Frame " << current_frame_idx << " | " << frame_count << " processed | " << (frame_count / elap) << " fps | Matches: " << match_count << endl;
        }
    }
    
    is_running = false;
    if (render_thread.joinable()) render_thread.join();
    cudaFree(d_map); cudaFree(d_frame); cudaFree(d_frame_f); cudaFree(d_map_f);
    cudaFree(d_out_render); cudaFree(d_pts1); cudaFree(d_pts2);
    return 0;
}
