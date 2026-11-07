#include "cuda_utils.h"
#include "save_mat.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <thread>
#include <mutex>
#include <queue>
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
    
    // Paths
    string video_path = "../GPSDenied/Videos/Digaru/CameraRecord/VID_EO_15.mp4";
    string map_path = "../map_data/reference_map_2km.png";
    string map_bin_path = "../map_data/reference_map_2km_features.bin";
    string ekf_path = "../sensor_data/VID_EO_15/ekf_state.csv";
    
    // Load TRT engines
    shared_ptr<ORB_SLAM3::SuperPoint> sp_drone_model = make_shared<ORB_SLAM3::SuperPoint>("../superpoint.engine");
    ORB_SLAM3::SPDetector sp_drone_det(sp_drone_model, false);
    
    shared_ptr<ORB_SLAM3::SuperPoint> sp_map_model = make_shared<ORB_SLAM3::SuperPoint>("../superpoint.engine");
    ORB_SLAM3::SPDetector sp_map_det(sp_map_model, false);

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
    
    VideoCapture cap(video_path);
    int start_frame = 79165;
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
    cudaMalloc(&d_map_f, 1024 * 1024 * sizeof(float));

    // Render thread setup
    std::mutex render_mutex;
    std::queue<Mat> render_queue;
    std::atomic<bool> is_running(true);
    std::thread render_thread([&]() {
        VideoWriter out("output_localizer.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 30, Size(2048, 1024));
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
        if (frame_count == 300) break; 

        int current_frame_idx = start_frame + frame_count;
        EKFState state = log_sync.getState(current_frame_idx, 59.94);
        auto t_frame_start = chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        auto t_cap_end = chrono::high_resolution_clock::now();

        cudaMemcpyAsync(d_frame, frame.data, frame.rows * frame.step, cudaMemcpyHostToDevice, stream);
        cuda_bgr2gray_resize_normalize(d_frame, frame.cols, frame.rows, frame.step, d_frame_f, 1024, 1024, stream);
        cudaStreamSynchronize(stream);
        auto t_pre_end = chrono::high_resolution_clock::now();
        
        // Launch drone detector asynchronously
        sp_drone_det.detectCUDA(d_frame_f, 1024, 1024);
        
        // --- Initialization Mode ---
        if (!has_locked_on) {
            // Need to wait for detection first
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
                    if(frame_count==130) cout << "Testing center " << c << ": " << centers[c].x << ", " << centers[c].y << endl;
                    cuda_crop_map_normalize(d_map, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, centers[c].x + pad, centers[c].y + pad, state.yaw * M_PI / 180.0f, stream);
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
if(frame_count==130) { save_device_float_mat(d_frame_f, 1024, 1024, "frame_f.jpg"); save_device_float_mat(d_map_f, 1024, 1024, "map_f.jpg"); }
                    cout << "Map kpts: " << map_kpts.size() << " Drone kpts: " << drone_kpts.size() << endl; auto lg_matches = lg_model->match(map_kpts, map_desc, drone_kpts, drone_desc, Size(1024, 1024));
                    int good_matches = lg_matches.size();
                    if (good_matches > best_matches) {
                        best_matches = good_matches;
                        center_x = centers[c].x;
                        center_y = centers[c].y;
                        match_count = good_matches;
                    }
                }
                if (best_matches >= 2) {
                    has_locked_on = true;
                    last_pn = state.pn;
                    last_pe = state.pe;
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
            cuda_crop_map_normalize(d_map, padded_map.cols, padded_map.rows, padded_map.step, d_map_f, 1024, 1024, center_x + pad, center_y + pad, state.yaw * M_PI / 180.0f, stream);
            cudaStreamSynchronize(stream);
            auto t_crop_end = chrono::high_resolution_clock::now();
            auto t_det_start = chrono::high_resolution_clock::now();
            // Launch Map async (Drone was already launched before the block!)
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

            auto t_match_start = chrono::high_resolution_clock::now();
            auto lg_matches = lg_model->match(map_kpts, map_desc, drone_kpts, drone_desc, Size(1024, 1024));
            match_count = lg_matches.size();
            auto t_match_end = chrono::high_resolution_clock::now();
            
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
    return 0;
}
struct PipelineData {
    int frame_idx;
    vector<KeyPoint> drone_kpts;
    vector<KeyPoint> map_kpts;
    Mat drone_desc;
    Mat map_desc;
    Mat frame;
    Mat map_crop;
    double center_x;
    double center_y;
    EKFState state;
};

std::queue<PipelineData> pipeline_queue;
std::mutex pipeline_mutex;
std::condition_variable pipeline_cv;
std::atomic<bool> pipeline_running(true);


