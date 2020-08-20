#!/bin/bash

touch prediction_output.txt

for f in cifar-10-batches-bin/*.bin; do
    ./predict models/fdeep_model_100_epochs.json $f 2>&1 | tee -a prediction_output.txt &
    echo | tee -a prediction_output.txt
    echo | tee -a prediction_output.txt
    echo -e "**************************\n" | tee -a prediction_output.txt
done

exit 0
