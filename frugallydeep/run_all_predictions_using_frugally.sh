#!/bin/bash
# runs sequentially by default
# to run predictions in parallel add argument 'parallel'

touch prediction_output.txt

for f in cifar-10-batches-bin/*.bin; do
    if [ $1 = 'parallel' ]; then
        ./predict models/fdeep_model_10_epochs.json $f &
    else
        ./predict models/fdeep_model_10_epochs.json $f
    fi
    echo
    echo
    echo -e "**************************\n"
done

exit 0
