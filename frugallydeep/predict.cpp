#include <fdeep/fdeep.hpp>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>

vector<float> readImageInColourOrder()
{
    // All red pixel values first, then green, then blue

    string filename = "cifar-10-batches-bin/data_batch_1.bin";
    ifstream dataFile(filename, std::ifstream::binary);
    char bucket;
    dataFile.read(&bucket, 1);
    float imageClass = (float)(uint8_t)bucket;
    std::cout << "image class " << imageClass << std::endl;

    // seek the first pixel value
    dataFile.seekg(1);

    char bytes[3072];
    dataFile.read(bytes, 3072);

    vector<float> image;

    float pixelValue;

    for (int i = 0; i < 3072; i++)
    {
        pixelValue = ((float)(uint8_t)bytes[i]) / 255;
        image.push_back(pixelValue);
    }

    image.push_back(imageClass);

    dataFile.close();

    return image;
}

vector<float> readImageAlternateColours()
{
    // return the image with red, green, blue pixel value of each pixel.
    vector<float> image = readImageInColourOrder();
    vector<float> newImage;
    for (int i = 0; i < 3072; i++)
    {
        int offset; //red values need no offset
        if (i % 3 == 0)
            offset = 0;
        else if (i % 3 == 2) // green values come after red
            offset = 1024;
        else if (i % 3 == 2) // blue values come last
            offset = 2048;

        newImage.push_back(image[i + offset]);
    }
    // After adding all pixels add the image class to end of vector
    newImage.push_back(image[image.size() - 1]);

    return newImage;
}

vector<float> softmax(vector<float> rawPredictions)
{
    vector<float> exponents;
    float sumExpo;

    for (int i = 0; i < rawPredictions.size(); i++)
    {
        exponents.push_back(exp(rawPredictions[i]));
        sumExpo += exponents[i];
    }

    vector<float> softmaxed;

    for (int i = 0; i < exponents.size(); i++)
    {
        softmaxed.push_back(exponents[i] / sumExpo);
    }

    return softmaxed;
}

int main()
{
    vector<float> image = readImageAlternateColours();
    float imageClass = image[3072];
    image.pop_back();

    const auto model = fdeep::load_model("fdeep_model.json");

    fdeep::tensors result = model.predict({fdeep::tensor(
        fdeep::tensor_shape(
            static_cast<std::size_t>(32),
            static_cast<std::size_t>(32),
            static_cast<std::size_t>(3)),
        image)});

    vector<float> t = result.at(0).to_vector();

    vector<float> probabilities = softmax(t);

    for (int i = 0; i < 10; i++)
    {
        std::cout << fixed << std::setprecision(29) << probabilities[i] << std::endl;
    }
}
