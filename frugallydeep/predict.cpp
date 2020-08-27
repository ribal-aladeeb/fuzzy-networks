#include <fdeep/fdeep.hpp>
#include <string>
using namespace std;
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>

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
        pixelValue = ((float)(uint8_t)bytes[i]) / 255.0;
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

vector<vector<float>> readAllImages(string filename = "cifar-10-batches-bin/data_batch_1.bin")
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
        // for (int j = 1; j < 3073; j++)
        for (int j = 1; j < 1025; j++)
        {
            image.push_back((float)(uint8_t)pixels[j] / 255.0);        // red value
            image.push_back((float)(uint8_t)pixels[j + 1024] / 255.0); // green value
            image.push_back((float)(uint8_t)pixels[j + 2048] / 255.0); // blue value
            // image.push_back(((float)(uint8_t)(pixels[j])) / 255);
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
    float maxRawPred = rawPredictions.at(0);
    for (int i = 1; i < rawPredictions.size(); i++)
    {
        if (maxRawPred < rawPredictions.at(i))
            maxRawPred = rawPredictions.at(i);
    }
    for (int i = 0; i < rawPredictions.size(); i++)
    {
        exponents.push_back(exp(rawPredictions[i] - maxRawPred));
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
vector<float> computeProbabilities(vector<float> &image, const fdeep::model &model)
{
    fdeep::tensors result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(32, 32, 3), image)});

    vector<float> rawLogits = result.at(0).to_vector();

    return softmax(rawLogits);
}

float calcAccuracy(string dataFile = "cifar-10-batches-bin/data_batch_1.bin",
                   string modelFile = "models/fdeep_model_100_epochs.json")
{
    vector<vector<float>> images = readAllImages(dataFile);
    int correctPredictions = 0;
    float totalPredictions = images.size();
    const auto model = fdeep::load_model(modelFile, false); // false to prevent fdeep sanity test (the test fails with MCA)

    for (int i = 0; i < images.size(); i++)
    {
        vector<float> image = images.at(i);
        int imageClass = (int)image.at(image.size() - 1); // the last index is where image class is stored
        image.pop_back();                                 // because frugally only needs pixel values for predictions

        // if (i == 0)
        // {
        //     std::cout << "Running all predictions currently" << std::endl;
        // }
        std::cout << "\rRunning prediction # " << i << std::flush;

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
    std::cout << std::endl
              << fixed << std::setprecision(16)
              << "number correct predictions " << correctPredictions
              << std::endl;
    return correctPredictions / totalPredictions;
}
void calcAccuracyDecorator(int argc, char **argv)
{
    string dataFilename, modelFilename;
    // bool verbose;
    if (argc >= 3)
    {
        modelFilename = argv[1];
        dataFilename = argv[2];
        // verbose = argv[3] == "--verbose" ? true : false;
        std::cout << "using model file " << modelFilename
                  //  << ", verbose: " << verbose << endl
                  << ", and data file " << dataFilename << endl;
        std::cout << fixed << std::setprecision(16) << calcAccuracy(dataFilename, modelFilename) << endl;
    }
    else
    {
        std::cout << fixed << std::setprecision(16) << calcAccuracy() << endl;
    }
}

int displayRGBinRowMajorOrder()
{
    string dataFilename = "cifar-10-batches-bin/data_batch_1.bin";
    vector<vector<float>> images = readAllImages(dataFilename);
    vector<float> image = images.at(0);
    image.pop_back();
    fdeep::tensor t = {fdeep::tensor(
        fdeep::tensor_shape(
            static_cast<std::size_t>(32),
            static_cast<std::size_t>(32),
            static_cast<std::size_t>(3)),
        image)};

    for (int y = 0; y < 32; y++)
    {
        for (int x = 0; x < 32; x++)
        {
            for (int z = 0; z < 3; z++)
            {
                int tmp = (int)(t.get(fdeep::tensor_pos(y, x, z)) * 255);
                cout << tmp << " ";
            }
            cout << std::endl;
        }
    }
    cout << fdeep::show_tensor(t) << std::endl;
}

int *runMCASingleImage(fdeep::model &model, vector<float> image, int trials)
{ // This function runs multiple inferences on the same image and uses MCA to
    // find out the number of significant digits in the prediction.
    vector<vector<float>> results;

    float sums[10]; // because there are 10 classes to cifar10
    for (int i = 0; i < trials; i++)
    {
        cout << "\r\tcomputing probabilities trial " << i << std::flush;
        vector<float>
            probs = computeProbabilities(image, model);
        for (int classes = 0; classes < probs.size(); classes++)
        {
            sums[classes] += probs[classes];
        }
        results.push_back(probs);
    }
    float means[10];
    float sigmas[10];
    int sigfigs[10];
    cout << std::endl;
    for (int i = 0; i < 10; i++)
    {
        means[i] = sums[i] / trials;
        float SumOfxMinusMeansSquared;
        // cout << "computing means and sigmas for class " << i << std::endl;
        for (int j = 0; j < trials; j++)
        {
            SumOfxMinusMeansSquared += pow(results.at(j).at(i) - means[i], 2);
        }
        sigmas[i] = sqrt(SumOfxMinusMeansSquared / trials);
        sigfigs[i] = -log10(sigmas[i] / abs(means[i]) + 1e-16);
        cout << "\t\tFor class " << i
             << " mean: " << means[i]
             << " standard dev: " << sigmas[i]
             << " significant figures: " << sigfigs[i] << "\n";
    }
    return sigfigs;
}
vector<int> find10images(vector<vector<float>> images)
{
    // this function returns the indices of the first image of each class. There
    // are 10 classes so 10 indices are returned.

    vector<int> indices(10, 0);
    bool seenThisClass[10] = {false};

    //because you already have image 0 at index 0, mark the first image's class as seen
    seenThisClass[(int)images.at(0).at(3072)] = true;

    for (int i = 1, j = 1; j < 10; i++)
    {
        int imageClass = images.at(i).at(3072);
        if (seenThisClass[imageClass] == false)
        {
            indices[j] = i;
            seenThisClass[imageClass] = true;
            j += 1;
        }
    }

    return indices;
}

void MCAexperiment(int trials = 30,
                   int numImages = 5,
                   string testFile = "cifar-10-batches-bin/test_batch.bin",
                   string modelFilename = "models/mobilenetv2_15_epochs.json")
{
    vector<vector<float>> images = readAllImages(testFile);
    vector<int> indices = find10images(images);
    fdeep::model model = fdeep::load_model(modelFilename, false);
    for (int i = 0; i < indices.size() && i < 5; i++)
    {
        vector<float> targetImage = images.at(indices[i]);
        float imageClass = targetImage.at(3072);
        targetImage.pop_back();
        cout << "Running Experiment on image index " << indices[i]
             << " of class " << imageClass
             << " from the test set\n";
        runMCASingleImage(model, targetImage, trials);
    }
}

int main(int argc, char **argv)
{
    MCAexperiment();
}
