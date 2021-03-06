FROM verificarlo/verificarlo

WORKDIR /new-workdir
# install frugally-deep dependencies
RUN apt-get update \
    && apt-get install -y build-essential cmake vim clang \
    && git clone -b 'v0.2.8-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus \
    && cd FunctionalPlus \
    && mkdir -p build && cd build \
    && cmake .. \
    && make && make install \
    && cd ../.. \
    && git clone -b '3.3.7' --single-branch --depth 1 https://github.com/eigenteam/eigen-git-mirror \
    && cd eigen-git-mirror \
    && mkdir -p build && cd build \
    && cmake .. \
    && make && make install \
    && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen \
    && cd ../.. \
    && git clone -b 'v3.7.3' --single-branch --depth 1 https://github.com/nlohmann/json \
    && cd json \
    && mkdir -p build && cd build \
    && cmake -DBUILD_TESTING=OFF .. \
    && make && make install \
    && cd ../.. \
    && git clone https://github.com/Dobiasd/frugally-deep \
    && cd frugally-deep \
    && mkdir -p build && cd build \
    && cmake .. \
    && make && make install \
    && cd ../.. 

COPY models/ models/
COPY cifar-10-batches-bin/ cifar-10-batches-bin/
COPY predict.cpp predict.cpp
RUN verificarlo-c++ predict.cpp -o predict
# RUN verificarlo-c++ --verbose predict.cpp -o predict

CMD ["bash","-c", "VFC_BACKENDS='libinterflop_mca.so --precision-binary32 24 --precision-binary64 53 --mode=mca' ./predict"]
#  CMD ["bash","-c", "VFC_BACKENDS='libinterflop_ieee.so --debug' ./predict"]

