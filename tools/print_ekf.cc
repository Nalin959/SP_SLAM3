#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

struct EKFState {
    double time;
    double pn, pe;
    double yaw;
};

int main() {
    ifstream file("../sensor_data/VID_EO_15/ekf_state.csv");
    string line, cell;
    bool first_line = true;
    int time_idx = -1, pn_idx = -1, pe_idx = -1, yaw_idx = -1;
    vector<EKFState> states;
    bool first_yaw = true;
    double prev_yaw = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;
        while (getline(ss, cell, ',')) row.push_back(cell);
        if (row.empty()) continue;
        if (first_line) {
            for (size_t i=0; i<row.size(); ++i) {
                if (row[i]=="timestamp_us") time_idx=i;
                else if (row[i]=="pn") pn_idx=i;
                else if (row[i]=="pe") pe_idx=i;
                else if (row[i]=="yaw") yaw_idx=i;
            }
            first_line=false; continue;
        }
        EKFState s;
        static double first_timestamp = -1;
        double ts = stod(row[time_idx]);
        if (first_timestamp < 0) first_timestamp = ts;
        s.time = (ts - first_timestamp) / 1e6;
        s.pn = stod(row[pn_idx]);
        s.pe = stod(row[pe_idx]);
        double raw_yaw = stod(row[yaw_idx]);
        if (first_yaw) { s.yaw = raw_yaw; first_yaw = false; }
        else {
            double diff = raw_yaw - prev_yaw;
            while (diff > 180.0) diff -= 360.0;
            while (diff < -180.0) diff += 360.0;
            s.yaw = prev_yaw + diff;
        }
        prev_yaw = s.yaw;
        states.push_back(s);
    }
    double t = 79165 / 59.94;
    auto it = lower_bound(states.begin(), states.end(), t, [](const EKFState& s, double val) {
        return s.time < val;
    });
    cout << "Center PN: " << it->pn << " PE: " << it->pe << " Yaw: " << it->yaw << endl;
}
