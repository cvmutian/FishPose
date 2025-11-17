## Environment Setup

### Prerequisites
- Anaconda or Miniconda
- NVIDIA GPU with CUDA 12.1 or compatible version
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/cvmutian/FishPose.git
   cd FishPose
   ```

2. **Create and Activate Conda Environment**
   ```bash
   conda create -n fishpose python=3.10 -y
   conda activate fishpose
   ```

3. **Install PyTorch**
   
   For CUDA 12.x:
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset can be downloaded via the following link:[FishPose_dataset](https://pan.baidu.com/s/1N8fJRRQL7UEWd5RvdWQKaw)(code:bt6s)

The dataset should be organized as follows:
```
FishPose/
└── dataset/
   ├── annotations/
       └── ...
   └── train/
       └── ...
   └── test/
       └── ...
```

## Model Weights

The model can be downloaded via the following link:[best_model.pth](https://pan.baidu.com/s/1HbIGgOWynwhG_thXTuU6Hw)(code:4tyv)

Make sure you have the model weights file placed in the `weights/` directory:
```
weights/
└── best_model.pth
```

## Usage

### Evaluation

Run evaluation on the test set:
```bash
python test.py 

```

