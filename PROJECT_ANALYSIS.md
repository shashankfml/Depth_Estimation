# SPAD Depth Estimation Project Analysis

## Analysis Summary

Based on the examination of the Jupyter notebooks, training logs, loss plots, and competition PDF, this project focuses on **monocular depth estimation from SPAD (Single Photon Avalanche Diode) camera images**.

### Key Findings from Analysis

#### 1. **Competition Context** (from EE5178-Course Competition.pdf)
- **Challenge**: Depth estimation from SPAD images under extreme conditions
- **SPAD Advantages**: 
  - Single photon detection capability
  - Up to 100,000 FPS temporal resolution
  - Excellent for low-light, high dynamic range, motion blur scenarios
  - Direct photon counting eliminates read noise

#### 2. **Model Architectures Implemented**
From the notebooks and existing code, three main approaches were identified:

**a) U-Net Architecture**
- Enhanced with batch normalization
- Skip connections for spatial information preservation
- Encoder-decoder structure with bilinear upsampling

**b) Multi-Scale Deep Network (Eigen et al.)**
- Two-stage approach: Coarse + Fine networks
- Coarse network predicts overall scene depth
- Fine network refines predictions locally
- Based on NIPS 2014 paper

**c) Custom DepthNet with Layer Normalization**
- **Best performing model** based on training logs
- Layer normalization instead of batch normalization
- U-Net style skip connections
- Achieved best validation loss of ~0.035

#### 3. **Training Results Analysis**
From the training logs and loss plots:

**Best Configuration:**
- Model: DepthNetGray (Custom architecture)
- Optimizer: SGD with momentum=0.9, lr=0.01
- Loss Function: MSE Loss
- Best Validation Loss: 0.0353 (epoch 28)
- Training showed good convergence without overfitting

**Key Insights:**
- Layer normalization outperformed batch normalization
- SGD with momentum was more effective than Adam
- Skip connections were crucial for spatial information preservation
- Training converged rapidly in first 5 epochs, then steady improvement

#### 4. **Performance Metrics**
- Primary metric: MSE Loss
- Best validation loss: ~0.035
- Good generalization (train/val losses converged)
- Stable training across multiple runs

## Refactored Implementation

### Project Structure
```
├── config.py          # Centralized configuration
├── dataset.py         # SPAD dataset handling
├── models.py          # All three architectures
├── train.py           # Training utilities with Trainer class
├── utils.py           # Helper functions and metrics
├── main.py            # Main training script
├── evaluate.py        # Model evaluation and visualization
├── demo.py            # Demo script for testing
└── README.md          # Comprehensive documentation
```

### Key Improvements Made

1. **Modular Design**: Separated concerns into distinct modules
2. **Configuration Management**: Centralized settings in config.py
3. **Trainer Class**: Encapsulated training logic with logging and checkpointing
4. **Multiple Architectures**: Implemented all three approaches in models.py
5. **Evaluation Framework**: Comprehensive metrics and visualization tools
6. **Documentation**: Detailed README with research references
7. **Experiment Tracking**: Organized output structure for reproducibility

### Research References Added

The README includes 8 key research papers covering:
- **Core Depth Estimation**: Eigen et al. (NIPS 2014), U-Net (MICCAI 2015)
- **SPAD Technology**: Recent papers on single-photon imaging applications
- **Survey Papers**: Comprehensive reviews of monocular depth estimation
- **Loss Functions**: Advanced techniques for depth estimation optimization

### Usage Examples

**Training:**
```bash
python main.py --model depthnet --epochs 30 --batch_size 8
python main.py --model unet --lr 0.001 --experiment_name unet_experiment
```

**Evaluation:**
```bash
python evaluate.py --model depthnet --checkpoint saved_models/best_model.pth --visualize
```

### Technical Specifications

- **Input**: Grayscale SPAD images (256×256)
- **Output**: Depth maps (256×256)
- **Best Model**: DepthNetGray with layer normalization
- **Optimizer**: SGD (lr=0.01, momentum=0.9, weight_decay=1e-8)
- **Loss**: MSE Loss
- **Performance**: Validation loss ~0.035

## Conclusion

The refactored codebase provides a clean, modular implementation of the SPAD depth estimation project with:
- Three different model architectures
- Comprehensive training and evaluation framework
- Proper documentation and research context
- Reproducible experiment structure
- Based on actual performance analysis from the original notebooks

The implementation is ready for further experimentation and can serve as a solid foundation for SPAD-based depth estimation research.
