#include <fdeep/fdeep.hpp>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>


vector<float> readImageInColourOrder() {
    // All red pixel values first, then green, then blue

    string filename = "cifar-10-batches-bin/data_batch_1.bin";
    ifstream dataFile(filename, std::ifstream::binary);
    char bucket;
    dataFile.read(&bucket, 1);
    float imageClass = (float)(uint8_t)bucket;
    std::cout <<"image class "<< imageClass << std::endl;
    
    // seek the first pixel value
    dataFile.seekg(1);
    
    char bytes[3072];
    dataFile.read(bytes, 3072);

    vector<float> image;

    float pixelValue;

    for (int i=0; i<3072; i++)
    {
        pixelValue = ((float)(uint8_t)bytes[i])/255;
        image.push_back(pixelValue);
    }

    image.push_back(imageClass);

    dataFile.close();

    return image;
}

// TODO
// vector<float> readImageAlternateColours(){
//     // red, green, blue pixel value of each pixel
//     return null
// }


vector<float> generate1Dimage()
{
    vector<float> image(32 * 32 * 3);
    for (int i = 0; i < image.size(); i++)
    {
        image[i] = std::rand() % 255 / 255.0;
    }
    return image;
}

int main()
{
    vector<float> image = readImageInColourOrder();
    float imageClass  = image[3073];
    image.pop_back();
    cout << imageClass<<"__"<< image.size() <<std::endl;

    const auto model = fdeep::load_model("fdeep_model.json");

    const auto result = model.predict({ fdeep::tensor(fdeep::tensor_shape(
        static_cast<std::size_t>(32), static_cast<std::size_t>(32), static_cast<std::size_t>(3)), image) });
    std::cout << std::setprecision(16); // show 16 digits of precision
    std::cout << fdeep::show_tensors(result) << std::endl;
}
