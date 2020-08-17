read-od: 
	od -t u1 cifar-10-batches-bin/data_batch_1.bin | head

compile:
	g++ --std=c++14 predict.cpp -o forward

run: 
	./forward

buildrun: compile run

docker-build:
	docker build . -t fuzzy-frugally

docker-build-no-cache:
	docker build . -t fuzzy-frugally --no-cache

docker-run:
	docker run --rm -it fuzzy-frugally
