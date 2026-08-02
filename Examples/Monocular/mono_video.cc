#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<System.h>

using namespace std;

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_video path_to_vocabulary path_to_settings path_to_video" << endl;
        return 1;
    }

    string vocab_path = argv[1];
    string settings_path = argv[2];
    string video_path = argv[3];

    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, true);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Failed to open video file!" << endl;
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps == 0.0) fps = 30.0;
    
    int frame_count = 0;
    double total_time = 0;

    cout << endl << "-------" << endl;
    cout << "Start processing video ..." << endl;

    while(true)
    {
        cv::Mat im;
        cap >> im;
        if(im.empty()) break;
        
        cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
        cv::resize(im, im, cv::Size(640, 480));

        double tframe = chrono::duration_cast<chrono::duration<double>>(
                            chrono::steady_clock::now().time_since_epoch()).count();

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe);
        
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        total_time += ttrack;
        frame_count++;
        
        // Wait to load the next frame
        double T = 1.0 / fps;
        if(ttrack < T)
            usleep((T - ttrack)*1e6);
    }
    
    cout << "Average processing time per frame: " << (total_time / frame_count) * 1000 << " ms (" << frame_count / total_time << " FPS)" << endl;

    // Stop all threads
    SLAM.Shutdown();
    
    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
