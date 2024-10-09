# The SapBERT Training Pipeline

This directory houses the SapBERT Training Pipeline. It is heavily adapted from the [SapBERT repository](https://github.com/cambridgeltl/sapbert), and all credits on the design of the training process go to the SapBERT authors. We've made the following modifications to the original code:

- Added evaluation on an epoch by epoch basis in the `train/train.py` script, in order to assess the best epoch during training, and keep the best model.
- Refactored the `evaluation/utils.py` script to compute the predictions, accuracy, and validation loss in evaluation time (to accomodate the previous change).
- Added code to structure the outputs in a way that aligns with our Entity Linking pipeline.
- Just made the code more readable and concise throughout, by removing a lot of code we didn't use in the aforementioned files, and also deleting unused files completely.

The key point is: **we did not alter the main logic of the training process in any way**. Check out the official repository for more information on the whole pipeline.
