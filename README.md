# 24S2-SC4001 Fashion-MNIST Classification

Convolutional neural network study on Fashion-MNIST, submitted for SC4001
(CE/CZ4042) Neural Networks and Deep Learning at Nanyang Technological
University.

## Overview

The project trains four CNN configurations on the same data split and compares
them on accuracy, parameter count, and interpretability. Three are distinct
architectures and the fourth is an augmentation variant. The comparison
isolates one variable at a time: a baseline, the same depth with dilated
convolutions, a smaller depthwise separable design, and the baseline again
trained with MixUp augmentation.

## Results

Test accuracy on the 10000 image Fashion-MNIST test set:

| Model | Parameters | Test Accuracy | Note |
| --- | --- | --- | --- |
| Baseline CNN | 155,850 | 92.69% | Reference architecture |
| Dilated CNN | 155,850 | 92.30% | Dilated convolutions, 0.39 points below baseline |
| Efficient CNN | 74,890 | 91.96% | 51.9% fewer parameters, 0.73 points below baseline |
| Baseline CNN with MixUp | 155,850 | 93.19% | Best result, 0.50 points above baseline |

MixUp augmentation produced the best model. Widening the receptive field with
dilated convolutions and shrinking the network with a depthwise separable
design both cost a small amount of accuracy, with the efficient model giving up
0.73 points for roughly half the parameters.

How to read these figures: each model trains for 10 epochs and is evaluated on
the test set after every epoch, and the table reports the best epoch. The
notebook holds out no separate validation split, so the test set also selects
the checkpoint and the accuracies are slightly optimistic for that reason.

## Repository Structure

| Path | Contents |
| --- | --- |
| `24S2-SC4001-Fashion-MNIST-Classification.ipynb` | Complete study in 22 sections, from data loading through interpretability analysis |
| `models/fashion_mnist_best_model.pth` | Trained weights for the best model, Baseline CNN with MixUp |
| `models/fashion_mnist_best_model_info.txt` | Architecture summary and parameter count for the saved model |
| `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz` | Fashion-MNIST training split, 60000 images |
| `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` | Fashion-MNIST test split, 10000 images |

## Data

Fashion-MNIST, 28 by 28 grayscale images across 10 clothing classes, 60000
training and 10000 test examples. The dataset files are tracked in this
repository, so the notebook reads them directly and does not need a network
connection. If the local files are missing it falls back to the raw copies on
GitHub, and then to the torchvision distribution.

Note that the torchvision fallback applies a different normalization. The local
and GitHub paths scale pixels to the range 0 to 1, which is what produced the
results above, while the torchvision path normalizes to the range -1 to 1.
Prefer the tracked files when reproducing the numbers.

### Data Source and License

The four `.gz` archives are unmodified copies of the Fashion-MNIST files
published by Zalando Research at
[zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).
Their MD5 checksums match the published ones. The dataset is distributed under
the MIT License, Copyright © 2017 Zalando SE. The full notice is reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Citation: Han Xiao, Kashif Rasul and Roland Vollgraf. Fashion-MNIST: a Novel
Image Dataset for Benchmarking Machine Learning Algorithms.
[arXiv:1708.07747](https://arxiv.org/abs/1708.07747).

## What the Notebook Covers

- Data loading, preprocessing, and dataset visualization.
- Baseline CNN with batch normalization and max pooling.
- Dilated convolution variant, with a receptive field comparison.
- Depthwise separable architecture inspired by MobileNet.
- MixUp augmentation, including a visualization of mixed examples.
- Model comparison across accuracy and parameter count.
- Error analysis on misclassified examples.
- Feature space visualization with t-SNE.
- Parameter efficiency against performance.
- Class activation maps for interpretability.
- Statistical analysis of the performance evaluation.

## Requirements

- Python 3
- torch and torchvision
- numpy, scipy, scikit-learn, scikit-image
- matplotlib, seaborn
- tqdm, requests

```bash
pip install torch torchvision numpy scipy scikit-learn scikit-image matplotlib seaborn tqdm requests
```

The notebook selects CUDA when available and otherwise runs on CPU. Training
four models on CPU takes considerably longer than on a GPU.

## Running the Analysis

```bash
git clone https://github.com/benjaminyjr19/24S2-SC4001-Fashion-MNIST-Classification.git
cd 24S2-SC4001-Fashion-MNIST-Classification
jupyter notebook 24S2-SC4001-Fashion-MNIST-Classification.ipynb
```

Execute the cells in order. Section 22 writes the best model to the `models`
directory, guarded by existence checks.

## Reproducibility

Section 1 calls `set_seed(42)`, which seeds NumPy and PyTorch, seeds CUDA when
a device is present, and sets `torch.backends.cudnn.deterministic` to true with
`benchmark` disabled. Stored cell outputs come from the original run on Python
3.12.7. Reproducing the exact figures also depends on matching library versions
and the same data loading path, since the torchvision fallback normalizes
differently.

## License

The notebook, the saved model and this documentation are under the MIT License.
See [LICENSE](LICENSE). The Fashion-MNIST archives remain under Zalando SE's own
MIT license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
