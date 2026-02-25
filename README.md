
# 💫 BBNN – Building Basic Neural Networks

**BBNN** is a lightweight C++ framework for constructing and training fully connected feed-forward neural networks with **heterogeneous activation functions**. Unlike most neural network libraries, BBNN allows activation functions to be assigned **per layer or per node**, enabling fine-grained experimentation with network structure.

BBNN is designed for **research, experimentation, and HPC training workflows**, with a focus on flexibility, reproducibility, and performance. This framework is intended for research and learning basic ANN structure and backpropagation. 

---

## 🗝️ Key Features

* Fully-connected feed-forward neural networks.
* Arbitrary number of hidden layers.
* Custom node counts per layer.
* **Mixed activation functions within layers.**
* Layer-level and node-level activation control.
* Adam optimizer.
* Deterministic seeds.
* CSV dataset input.
* CSV model output.
* HPC-friendly batch execution.

---

## 🗃️ Architecture Overview

BBNN constructs standard feed-forward neural networks:

```
Input Layer -> Hidden Layer 1 -> Hidden Layer 2 -> ... -> Output Layer
```

Example configuration:

```
hNodes 2 50 50
```

Produces:

```
Input -> 50 -> 50 -> Output
```

### Network Diagram

```
            Input Features
                  │
                  ▼
        ┌───────────────────┐
        │   Hidden Layer 1  │
        │   50 Nodes        │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Hidden Layer 2  │
        │   50 Nodes        │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Output Layer    │
        │   Softmax         │
        └───────────────────┘
```

---

## 🟡 Activation Functions

### ReLU Family

| ID | Function       |
| -- | -------------- |
| 0  | ReLU (default) |
| 1  | ELU            |
| 2  | Leaky ReLU     |
| 3  | GeLU           |
| 4  | Swish          |

### Sigmoid Family

| ID | Function        |
| -- | --------------- |
| 5  | Sigmoid         |
| 6  | Bipolar Sigmoid |
| 7  | Tanh            |

### Gaussian

| ID | Function |
| -- | -------- |
| 8  | Gaussian |

### Output Functions

| ID | Function           |
| -- | ------------------ |
| 9  | Softmax (Training) |
| 10 | Argmax (Testing)   |

### Recommended Output Layers

```
Hidden Layer ->  Sigmoid/Tanh -> Softmax
```

### Heterogeneous Activation Functions

BBNN supports assigning activation functions at **multiple levels**:

* Global default
* Per-layer
* Per-node
* Randomized distributions

This enables architectures where:

```
Layer 1:

Node:  1  2  3  4  5  6
Act:   R  R  T  T  G  S
```

* R = ReLU
* T = Tanh
* G = Gaussian
* S = Swish

---

## 🏗️ Building

Compile using:

```bash
make clean
make
```

Executable:

```
build/main
```

---

## 👟 Running

Basic usage:

```bash
./main <dataset.csv> [options]
```

Example:

```bash
./main ../data/mnist.csv \
  LogPath ../results/test/log.csv \
  ANN_Path ../results/test/ann.csv \
  Adam alpha 0.001 maxIter 10000 \
  hNodes 2 50 50
```

---

## 📊 Dataset Format

Datasets must be CSV files.

Default behavior:

| Parameter       | Default |
| --------------- | ------- |
| Class ID Column | 0       |
| Rows Skipped    | 1       |
| Columns Skipped | 0       |

Options:

```
ID_column x
skip_row x
skip_column x
```

---

## 🌐 Network Configuration

### Hidden Layers

```
hNodes x y1 y2 ... yn
```

Example:

```
hNodes 3 32 64 32
```

Produces:

```
Input -> 32 -> 64 -> 32 -> Output
```

---

## 🟢 Activation Configuration

### Default Activation

```
set_actDefault x
```

Example:

```
set_actDefault 7
```

Sets all hidden layers to **Tanh**.



### Random Activations

```
aseed x
```

Custom list:

```
aseed x list: y1 y2 ... yn :list
```

Example:

```
aseed 10 list: 0 7 8 :list
```

Randomly selects:

* ReLU
* Tanh
* Gaussian



### Per-Layer Activation

```
set_actLayer <actID> <layer>
```

Example:

```
set_actLayer 7 2
```

Layer 2 -> Tanh



### Multiple Layers

Range:

```
set_actLayers x y1 y2
```

List:

```
set_actLayers x list: y1 y2 ... yn :list
```



### Per-Node Activation

Range:

```
set_actNodes <actID> <startNode> <count>
```

Example:

```
set_actNodes 8 0 10
```

First 10 nodes → Gaussian.

List:

```
set_actNodes x list: y1 y2 ... yn :list
```

---

## 🔄️ Training Options

### Iterations

```
maxIter x
```

Default:

```
1000
```



### Train/Test Split

```
ratio x
```

Default:

```
70
```

Meaning:

```
70% Training
30% Testing
```



### Random Seeds

```
sseed x     # dataset sampling
wseed x     # weight initialization
aseed x     # activation selection
```

---

## ⏫ Optimizer

### Adam

Enable:

```
Adam
```

Learning rate:

```
alpha x
```

Defaults:

```
alpha 0.01
beta 0.9 0.99
```

Set beta parameters:

```
beta x y
```

---

## ➡️ Output Files

### Training Log

```
LogPath <file>
```

Default:

```
../results/log.csv
```

Contains:

* Training progress
* Runtime statistics


### Network Parameters

```
ANN_Path <file>
```

Default:

```
../results/ann.csv
```

Contains:

* Network weights
* Network structure
* Activation assignments

---

## 🚀 Example HPC Job Script

Example SLURM job:

```bash
./main ../data/mnist.csv \
  LogPath ../results/test/log.csv \
  ANN_Path ../results/test/ann.csv \
  Adam alpha 0.001 maxIter 10000 \
  hNodes 2 50 50
```

---

## 🧪 Research Applications

BBNN is particularly suited for:

* Activation function studies
* Neural architecture experiments
* Randomized networks
* Evolutionary neural networks
* Small-to-medium datasets
* HPC batch training

It is intended as a **research-oriented neural network toolkit** rather than a large-scale deep learning framework.

### Citing This Work
If you use this package in academic publications, please cite:
```
@software{Escarzaga_BBNN_2024,  
  author = {Escarzaga, Felicity},  
  month = Apr,  
  title = {{Building Basic Neural Networks (BBNN)}},   
  url = {https://github.com/FelicityE/BBNN},  
  version = {3.0},  
  year = {2024}  
}
```

---
## 📬 Contact
Developed by **Felicity E.** For questions or collaborations, please reach out via GitHub issues.
