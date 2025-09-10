"""
Demo script to test the SPAD depth estimation models.
"""

import torch
import numpy as np
from models import get_model
from utils import count_parameters, set_seed


def test_models():
    """Test all model architectures."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = ['unet', 'multiscale', 'depthnet']
    
    # Create dummy input (batch_size=2, channels=1, height=256, width=256)
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    
    print("Testing SPAD Depth Estimation Models")
    print("=" * 50)
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()} model:")
        print("-" * 30)
        
        try:
            # Create model
            model = get_model(model_name).to(device)
            
            # Count parameters
            param_info = count_parameters(model)
            print(f"Total parameters: {param_info['total']:,}")
            print(f"Trainable parameters: {param_info['trainable']:,}")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                if model_name == 'multiscale':
                    # Multi-scale model returns two outputs
                    coarse_output, fine_output = model(dummy_input)
                    print(f"Coarse output shape: {coarse_output.shape}")
                    print(f"Fine output shape: {fine_output.shape}")
                else:
                    output = model(dummy_input)
                    print(f"Output shape: {output.shape}")
            
            print("✓ Model test passed!")
            
        except Exception as e:
            print(f"✗ Model test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Model testing completed!")


def test_data_flow():
    """Test data loading and preprocessing."""
    print("\nTesting data flow...")
    
    try:
        from dataset import get_transforms
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode='L')
        
        # Test transforms
        transform = get_transforms()
        transformed = transform(dummy_img)
        
        print(f"Original image size: {dummy_img.size}")
        print(f"Transformed tensor shape: {transformed.shape}")
        print("✓ Data flow test passed!")
        
    except Exception as e:
        print(f"✗ Data flow test failed: {str(e)}")


if __name__ == "__main__":
    print("SPAD Depth Estimation - Demo Script")
    print("=" * 50)
    
    # Test models
    test_models()
    
    # Test data flow
    test_data_flow()
    
    print("\nDemo completed successfully!")
    print("\nTo train a model, run:")
    print("python main.py --model unet --epochs 5")
    print("\nTo evaluate a model, run:")
    print("python evaluate.py --model unet --checkpoint path/to/model.pth")
