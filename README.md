# Bean Leaf Classification with Transfer Learning
### Custom CNN vs GoogLeNet in PyTorch

My first **transfer learning project in PyTorch**, comparing a custom CNN trained from scratch with two approaches using pretrained GoogLeNet:

- **Full fine-tuning:** updating all model parameters.
- **Frozen backbone parameters:** training a new classifier while keeping pretrained parameters fixed.

All three models use a shared data pipeline.

## Project Objectives

- Build a custom CNN as a baseline.
- Adapt pretrained GoogLeNet to a three-class classification task.
- Understand parameter freezing and full fine-tuning.
- Compare training and validation accuracy and loss.
- Examine how different training strategies affect model performance.

## Dataset

The project uses the [Bean Leaf Lesions Classification dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification) from Kaggle.

| Property | Value |
|---|---:|
| Training images, according to the CSV | 1,034 |
| Validation images, according to the CSV | 133 |
| Number of classes | 3 |
| Input image size | 128 × 128 |
| Batch size | 32 |

Images are loaded using `torchvision.datasets.ImageFolder`.

> The notebook uses “test” in some variable names and plot labels, but the corresponding data comes from the dataset’s validation folder.

## Models Compared

| Model | Initialization | Trainable Parameters |
|---|---|---|
| Custom CNN | Random weights | Entire network |
| GoogLeNet — full fine-tuning | Pretrained backbone with a new classifier | Entire network |
| GoogLeNet — frozen backbone parameters | Pretrained backbone with a new classifier | Final classifier only |

In the frozen-parameter experiment, the full model remains in training mode during training. BatchNorm running statistics can therefore change even though the backbone parameters are frozen.

## Shared Preprocessing

The same preprocessing is used for all three models:

```python
transform.Compose([
    transform.Resize((128, 128)),
    transform.ToTensor(),
])
```

Training data is shuffled; validation data is not.

This version does not include data augmentation or input normalization. Its preprocessing differs from the standard preprocessing associated with pretrained GoogLeNet weights.

## Custom CNN Architecture

The baseline contains four convolutional blocks followed by a classifier.

| Stage | Operations | Output Shape |
|---|---|---|
| Input | RGB image | 3 × 128 × 128 |
| Block 1 | Conv2d → ReLU → MaxPool2d | 16 × 64 × 64 |
| Block 2 | Conv2d → ReLU → MaxPool2d | 32 × 32 × 32 |
| Block 3 | Conv2d → ReLU → MaxPool2d | 64 × 16 × 16 |
| Block 4 | Conv2d → ReLU → MaxPool2d | 128 × 8 × 8 |
| Classifier | Flatten → Linear → ReLU → Linear | 3 class scores |

The classifier maps **8,192 features → 128 hidden units → 3 outputs**.

## Training Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Pretrained model library | torchvision |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Epochs | 15 per model |
| Batch size | 32 |
| Device | CUDA when available; otherwise CPU |
| Execution environment | Google Colab |

Raw model outputs are used for loss calculation. Predicted classes are obtained using `argmax`.

## Results

The following values come from one saved experiment run.

**Metrics are equal-weight averages across batches, rather than exact per-image averages. These are validation results, not independent test scores.**

| Model | Final Training Accuracy | Final Validation Accuracy | Best Validation Accuracy | Best Epoch |
|---|---:|---:|---:|---:|
| Custom CNN | 93.09% | 80.63% | 82.50% | 9 |
| GoogLeNet — full fine-tuning | 98.86% | 93.75% | 98.13% | 13 |
| GoogLeNet — frozen backbone parameters | 86.33% | 88.13% | 88.13% | 15 |


### Observations

- Full fine-tuning achieved the highest recorded validation accuracy in this run.
- Training only the new classifier finished with a higher recorded validation accuracy than the custom CNN.
- The custom CNN showed a gap between training and validation performance, suggesting overfitting.
- The fine-tuned model reached its best validation score before the final epoch.

The custom CNN and GoogLeNet differ in architecture as well as initialization. Their comparison therefore does not isolate the effect of pretraining alone.

## Repository Structure

| Path | Description |
|---|---|
| `notebooks/bean_leaf_transfer_learning.ipynb` | Complete notebook with saved outputs |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## How to Run

### Google Colab

1. Open `notebooks/bean_leaf_transfer_learning.ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Start a fresh session and run the cells from top to bottom.
4. Provide your own Kaggle credentials when prompted.
5. Review the individual learning curves and model comparison plots.

The notebook downloads the dataset using `opendatasets` and uses paths under `/content/bean-leaf-lesions-classification/`.

### Local Jupyter

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Update the dataset paths to match your local download location before running the notebook.

The notebook redefines its training function for each experiment and uses global loaders and metric lists. Run it sequentially and recreate the models before starting a fresh experiment.

## What I Learned

- Building a CNN with PyTorch.
- Loading pretrained models with torchvision.
- Replacing a pretrained classification layer.
- Freezing parameters using `requires_grad`.
- Comparing full fine-tuning with classifier-only training.
- Separating training and validation phases.
- Recording and interpreting learning curves.
- Understanding how metric calculations affect reported results.

## Current Limitations

- Metrics give each batch equal weight, including smaller final batches.
- The validation split is inspected throughout training; no separate final test evaluation is included.
- Inputs do not use the standard pretrained GoogLeNet normalization.
- BatchNorm statistics still adapt in the frozen-parameter experiment.
- Results come from a single run without a fixed random seed.
- The notebook does not save the best model checkpoint.

## Future Improvements

- Calculate metrics using the total number of images.
- Experiment with normalization and data augmentation.
- Save the best validation checkpoint.
- Add confusion matrices and per-class precision, recall and F1-score.
- Repeat experiments using recorded random seeds.
- Evaluate the selected models on a held-out test set.
- Compare training time, inference speed and model size.

## Technologies Used

Python · PyTorch · torchvision · NumPy · Pandas · Matplotlib · scikit-learn · Google Colab

## Acknowledgments

- The Kaggle dataset and its original contributors.
- PyTorch and torchvision for the framework and pretrained GoogLeNet implementation.

Dataset files, credentials and pretrained weights are not included in this repository.

## Author

**Sifat Singh Bhatia**

[GitHub](https://github.com/Sifat192)
