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
#include <atomic>
#include <condition_variable>
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

// --- Pipeline Data Structures ---
struct PipelineData {
    int frame_idx;
    vector<KeyPoint> drone_kpts;
    vector<KeyPoint> map_kpts;
    Mat drone_desc;
    Mat map_desc;
    Mat frame;
    double center_x;
    double center_y;
    EKFState state;
};

std::queue<PipelineData> pipeline_queue;
std::mutex pipeline_mutex;
std::condition_variable pipeline_cv;
std::atomic<bool> pipeline_running(true);
std::atomic<int> backend_processed(0);


int main(int argc, char** argv) {
    cout << "Initializing Jetson Orin NX Video Localizer (C++ TRT)..." << endl;
    cout << "Make sure superpoint_dla.engine is compiled using: trtexec --onnx=superpoint.onnx --saveEngine=superpoint_dla.engine --useDLACore=0 --allowGPUFallback" << endl;

    // Paths
    string video_path = "../GPSDenied/Videos/Digaru/CameraRecord/VID_EO_15.mp4";
    string map_path = "../map_data/reference_map_2km.png";
    string map_bin_path = "../map_data/reference_map_2km_features.bin";
    string ekf_path = "../sensor_data/VID_EO_15/ekf_state.csv";
    
    // Load TRT engines. Drone and map detectors should use the DLA compiled engine!
    shared_ptr<ORB_SLAM3::SuperPoint> sp_drone_model = make_shared<ORB_SLAM3::SuperPoint>("../superpoint_dla.engine");
    ORB_SLAM3::SPDetector sp_drone_det(sp_drone_model, false);
    
    shared_ptr<ORB_SLAM3::SuperPoint> sp_map_model = make_shared<ORB_SLAM3::SuperPoint>("../superpoint_dla.engine");
    ORB_SLAM3::SPDetector sp_map_det(sp_map_model, false);

    // LightGlue runs on GPU
    shared_ptr<ORB_SLAM3::LightGlue> lg_model = make_shared<ORB_SLAM3::LightGlue>("../lightglue.engine");
    
    // Load EKF
    LogSynchronizer log_sync(ekf_path);
    
    // Load Map Features
    MapData global_map_features = loadMapDataBin(map_bin_path);
    cout << "Loaded global map features: " << global_map_features.kpts.size() << " kpts" << endl;
    
    // Use FlannBasedMatcher for global search
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    
    // Load Reference Map
    Mat ref_map = imread(map_path, IMREAD_GRAYSCALE);
    int LARGE_CROP = 1400;
    int pad = LARGE_CROP / 2;
    Mat padded_map;
    copyMakeBorder(ref_map, padded_map, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
    
    // Setup Zero-Copy Memory for the global padded map
    uint8_t* d_map = nullptr;
    // cudaHostAllocAllocates page-locked host memory that is accessible to the device.
    cudaHostAlloc((void**)&d_map, padded_map.total() * padded_map.elemSize(), cudaHostAllocMapped);
    memcpy(d_map, padded_map.data, padded_map.total() * padded_map.elemSize());
    
    // Device pointers for zero-copy
    uint8_t* d_map_ptr = nullptr;
    cudaHostGetDevicePointer((void**)&d_map_ptr, (void*)d_map, 0);

    // Use GStreamer pipeline for zero-copy hardware H.264 decoding (NVDEC)
    // Adjust format and appsink as needed for OpenCV 4 on Jetpack
    string gstreamer_pipeline = "filesrc location=" + video_path + " ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
    VideoCapture cap(gstreamer_pipeline, cv::CAP_GSTREAMER);
    
    if (!cap.isOpened()) {
        cout << "GStreamer pipeline failed! Falling back to standard FFmpeg backend for testing..." << endl;
        cap.open(video_path);
    }
    
    int start_frame = 79165;
    // Note: CAP_PROP_POS_FRAMES may not be perfectly accurate with GStreamer streaming.
    cap.set(CAP_PROP_POS_FRAMES, start_frame);
    
    Mat frame;
    bool has_locked_on = false;
    double center_x = 0, center_y = 0;
    double last_pn = 0, last_pe = 0;
    double MAP_SCALE = 0.534563;
    
    int frame_count = 0;
    auto start_time = chrono::high_resolution_clock::now();
    
    // Setup Zero-Copy Memory for input frame
    uint8_t* h_frame_buf = nullptr;
    int frame_size = 3840 * 2160 * 3; // Max size assumption
    cudaHostAlloc((void**)&h_frame_buf, frame_size, cudaHostAllocMapped);
    uint8_t* d_frame_ptr = nullptr;
    cudaHostGetDevicePointer((void**)&d_frame_ptr, (void*)h_frame_buf, 0);

    float* d_frame_f = nullptr;
    float* d_map_f = nullptr;
    cudaMalloc(&d_frame_f, 1024 * 1024 * sizeof(float));
    cudaMalloc(&d_map_f, 1024 * 1024 * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Backend Consumer Thread ---
    std::thread backend_thread([&]() {
        auto t_backend_start = chrono::high_resolution_clock::now();
        while (pipeline_running || !pipeline_queue.empty()) {
            PipelineData data;
            bool got_data = false;
            {
                std::unique_lock<std::mutex> lock(pipeline_mutex);
                pipeline_cv.wait_for(lock, std::chrono::milliseconds(5), [&]{ return !pipeline_queue.empty() || !pipeline_running; });
                if (!pipeline_queue.empty()) {
                    data = pipeline_queue.front();
                    pipeline_queue.pop();
                    got_data = true;
                }
            }

            if (got_data) {
                auto t_match_start = chrono::high_resolution_clock::now();
                // This utilizes GPU SMs (Orin GPU)
                auto lg_matches = lg_model->match(data.map_kpts, data.map_desc, data.drone_kpts, data.drone_desc, Size(1024, 1024));
                int match_count = lg_matches.size();
                auto t_match_end = chrono::high_resolution_clock::now();
                
                int b_processed = ++backend_processed;
                
                if (b_processed % 10 == 0) {
                    double match_time = chrono::duration<double, std::milli>(t_match_end - t_match_start).count();
                    auto t_now = chrono::high_resolution_clock::now();
                    double elapsed = chrono::duration<double>(t_now - t_backend_start).count();
                    cout << "  Backend Frame " << data.frame_idx << " | " << b_processed << " processed | " << (b_processed / elapsed) << " fps | Matches: " << match_count << " | Match GPU time: " << match_time << "ms" << endl;
                }
            }
        }
    });

    // --- Frontend Producer Loop ---
    while (cap.read(frame)) {
        if (frame_count == 300) break; 

        int current_frame_idx = start_frame + frame_count;
        EKFState state = log_sync.getState(current_frame_idx, 59.94);
        
        auto t_frame_start = chrono::high_resolution_clock::now();
        if (frame.empty()) {
            break;
        }
        auto t_cap_end = chrono::high_resolution_clock::now();

        // Zero copy optimization! Copy CPU mat into pinned memory buffer
        int current_size = frame.rows * frame.step;
        memcpy(h_frame_buf, frame.data, current_size);
        // Direct zero-copy normalize using the device pointer! (Bypasses cudaMemcpyAsync)
        cuda_bgr2gray_resize_normalize(d_frame_ptr, frame.cols, frame.rows, frame.step, d_frame_f, 1024, 1024, stream);
        cudaStreamSynchronize(stream);
        
        auto t_pre_end = chrono::high_resolution_clock::now();
        
        // Launch drone detector asynchronously on DLA
        sp_drone_det.detectCUDA(d_frame_f, 1024, 1024);
        
        if (!has_locked_on) {
            vector<KeyPoint> drone_kpts;
            sp_drone_det.getKeyPoints(0.005, 0, 1024, 0, 1024, drone_kpts, true);
            if (drone_kpts.size() > 4096) {
                sort(drone_kpts.begin(), drone_kpts.end(), [](const KeyPoint& a, const KeyPoint& b){ return a.response > b.response; });
                drone_kpts.resize(4096);
            }
            Mat drone_desc;
            sp_drone_det.computeDescriptors(drone_kpts, drone_desc);

            vector<vector<DMatch>> knn_matches;
            matcher->knnMatch(drone_desc, global_map_features.desc, knn_matches, 2);
            
            vector<Point2f> matched_map_pts;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < 0.85f * knn_matches[i][1].distance) {
                    matched_map_pts.push_back(global_map_features.kpts[knn_matches[i][0].trainIdx].pt);
                }
            }
            
            if (matched_map_pts.size() >= 10) {
                vector<int> labels;
                vector<Point2f> centers;
                simpleDBSCAN(matched_map_pts, 50.0f, 3, labels, centers);
                int best_matches = 0;
                for (size_t c = 0; c < min(centers.size(), size_t(3)); c++) {
                    // Direct zero-copy crop from pinned host memory!
                    cuda_crop_map_normalize(d_map_ptr, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, centers[c].x + pad, centers[c].y + pad, state.yaw * M_PI / 180.0f, stream);
                    cudaStreamSynchronize(stream);
                    sp_map_det.detectCUDA(d_map_f, 1024, 1024);
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
                    }
                }
                if (best_matches >= 2) {
                    has_locked_on = true;
                    last_pn = state.pn;
                    last_pe = state.pe;
                    cout << "LOCKED ON at center " << center_x << ", " << center_y << endl;
                }
            }
        } else {
            double delta_pe = state.pe - last_pe;
            double delta_pn = state.pn - last_pn;
            center_x += (delta_pe / MAP_SCALE);
            center_y += (-delta_pn / MAP_SCALE);
            last_pn = state.pn;
            last_pe = state.pe;
            
            auto t_crop_start = chrono::high_resolution_clock::now();
            cuda_crop_map_normalize(d_map_ptr, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, center_x + pad, center_y + pad, state.yaw * M_PI / 180.0f, stream);
            cudaStreamSynchronize(stream);
            auto t_crop_end = chrono::high_resolution_clock::now();
            
            auto t_det_start = chrono::high_resolution_clock::now();
            // Launch Map async on DLA
            sp_map_det.detectCUDA(d_map_f, 1024, 1024);
            
            // Now block and get results
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
            
            // Send to pipeline
            PipelineData data;
            data.frame_idx = current_frame_idx;
            data.drone_kpts = std::move(drone_kpts);
            data.map_kpts = std::move(map_kpts);
            data.drone_desc = std::move(drone_desc);
            data.map_desc = std::move(map_desc);
            data.center_x = center_x;
            data.center_y = center_y;
            data.state = state;
            
            {
                std::lock_guard<std::mutex> lock(pipeline_mutex);
                if (pipeline_queue.size() > 2) pipeline_queue.pop(); 
                pipeline_queue.push(data);
            }
            pipeline_cv.notify_one();
            
            if (frame_count % 10 == 0) {
                double crop_time = chrono::duration<double, std::milli>(t_crop_end - t_crop_start).count();
                double det_time = chrono::duration<double, std::milli>(t_det_end - t_det_start).count();
                double desc_time = chrono::duration<double, std::milli>(t_desc_end - t_desc_start).count();
                double pre_time = chrono::duration<double, std::milli>(t_pre_end - t_cap_end).count();
                double cap_time = chrono::duration<double, std::milli>(t_cap_end - t_frame_start).count();
                double total_loop_time = chrono::duration<double, std::milli>(t_desc_end - t_frame_start).count();
                cout << "  [FRONTEND TIMING] CapNVDEC: " << cap_time << "ms | ZC_Pre: " << pre_time << "ms | ZC_Crop: " << crop_time << "ms | DLA_Detect(Ovl): " << det_time << "ms | Desc: " << desc_time << "ms | Stage1Latency: " << total_loop_time << "ms" << endl;
            }
        }
        
        frame_count++;
        if (frame_count % 10 == 0) {
            auto end_time = chrono::high_resolution_clock::now();
            double elap = chrono::duration<double>(end_time - start_time).count();
            cout << "  Frontend | " << frame_count << " processed | " << (frame_count / elap) << " fps" << endl;
        }
    }
    
    pipeline_running = false;
    pipeline_cv.notify_all();
    backend_thread.join();
    
    cudaFreeHost(d_map); cudaFreeHost(h_frame_buf); 
    cudaFree(d_frame_f); cudaFree(d_map_f);
    return 0;
}
