#include <iostream>
#include "DBoW3.h"

int main(int argc, char **argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./convert_voc <input.yml.gz> <output.bin>" << std::endl;
        return 1;
    }

    std::string inFile = argv[1];
    std::string outFile = argv[2];

    std::cout << "Loading DBoW3 vocabulary from " << inFile << " (this will take 1-3 minutes)..." << std::endl;
    DBoW3::Vocabulary voc;
    voc.load(inFile);
    
    std::cout << "Vocabulary loaded successfully!" << std::endl;
    std::cout << "Saving to binary format: " << outFile << "..." << std::endl;
    
    voc.save(outFile);
    
    std::cout << "Saved successfully! You can now use the .bin file for instant loading." << std::endl;
    return 0;
}
