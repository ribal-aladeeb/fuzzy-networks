#include <fdeep/fdeep.hpp>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>

int readOneImage()
{
    string filename = "cifar-10-batches-bin/data_batch_1.bin";
    ifstream dataFile(filename, std::ifstream::binary);
    char bucket;
    dataFile.get(bucket);
    int imageClass = (int)bucket;
    std::cout << imageClass << std::endl;
    char line[32];
    dataFile.read(line, 32);
    float pixel[32];
    for (int i = 0; i < 32; i++)
    {
        pixel[i] = (float)line[i];
        std::cout << pixel[i] << std::endl;
    }
    dataFile.close();
}

vector<vector<vector<float>>> generateImage()
{

    vector<vector<vector<float>>> image(32, vector<vector<float>>(32, vector<float>(3, 0)));
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // this simulates the pixel value range used to train the model
                image[i][j][k] = float(std::rand() % 255 / 255.0);
            }
        }
    }
    return image;
}
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
    const auto image = generate1Dimage();
    const auto model = fdeep::load_model("fdeep_model.json");

    const auto result = model.predict({fdeep::tensor(fdeep::tensor_shape(
        static_cast<std::size_t>(32),static_cast<std::size_t>(32),static_cast<std::size_t>(3)), image)});
    std::cout << fdeep::show_tensors(result) << std::endl;
}
