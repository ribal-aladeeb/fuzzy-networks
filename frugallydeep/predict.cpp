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
    if (not dataFile.is_open())
    {
        std::cerr << "data file is not open" << std::endl;
        exit(1);
    }
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

vector<vector<float>> readAllImages(string &filename)
{ // This func will read all images given in the filename
    // in the same format as readImageAlternateColours
    vector<vector<float>> images;

    ifstream dataFile(filename, std::ifstream::binary);
    if (not dataFile.is_open())
    {
        std::cerr << "data file '" << filename << "' is not open" << std::endl;
        exit(1);
    }
    const int NUM_BYTES = 3073; // each file has 10k images, each image is 3073 bytes
    char pixels[NUM_BYTES];

    for (int i = 0; i < 10000; i++)
    {
        dataFile.read(pixels, NUM_BYTES);
        float imageClass = (float)(uint8_t)pixels[0];
        vector<float> image;
        for (int j = 1; j < 1025; j++)
        {
            image.push_back((float)(uint8_t)pixels[j]);        // red value
            image.push_back((float)(uint8_t)pixels[j + 1024]); // green value
            image.push_back((float)(uint8_t)pixels[j + 2048]); // blue value
        }

        image.push_back(imageClass);
        images.push_back(image);
    }
    return images;
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

int finalClassification(vector<float> probabilities)
{ // This function takes as input the output of softmax
    // it returns the index of the highest probability in the float vector
    int indexMax = 0;
    for (int i = 1; i < probabilities.size(); i++)
    {
        if (probabilities[indexMax] < probabilities[i])
        {
            indexMax = i;
        }
    }
    return indexMax;
}
int firstImageClassification()
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
        std::cout << fixed << std::setprecision(8) << probabilities[i] << std::endl;
    }
    cout << "image classified as " << finalClassification(probabilities) << std::endl;
}

float calcAccuracy()
{
    string fileName = "cifar-10-batches-bin/data_batch_1.bin";
    vector<vector<float>> images = readAllImages(fileName);
    int correctPredictions = 0;
    float totalPredictions = images.size();
    const auto model = fdeep::load_model("fdeep_model.json");

    for (int i = 0; i < images.size(); i++)
    {
        vector<float> image = images.at(i);
        int imageClass = (int)image.at(image.size() - 1); // the last index is where image class is stored
        image.pop_back();                                 // because frugally only needs pixel values for predictions

        cout << "Running prediction # " << i << std::endl;
        fdeep::tensors result = model.predict({fdeep::tensor(
            fdeep::tensor_shape(
                static_cast<std::size_t>(32),
                static_cast<std::size_t>(32),
                static_cast<std::size_t>(3)),
            image)});

        int predictedClass = finalClassification(softmax(result.at(0).to_vector()));

        if (predictedClass == imageClass)
            correctPredictions += 1;
    }
    cout << "number correct predictions " <<correctPredictions<< std::endl;
    return correctPredictions / totalPredictions;
}
int main()
{

    cout << "The accuracy is " << calcAccuracy() << std::endl;
}