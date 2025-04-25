import torch
import torch.nn as nn
from torchvision.models import VisionTransformer
from torchvision import transforms
from PIL import Image, ImageFile
import os
import sys

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
config = {
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "flowchart_vit.pth",
    "class_names": ['non_flowchart', 'flowchart'],
    "images_folder": "./dataset/val/flowchart"  # Hardcoded path
}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=2
    )
    model.load_state_dict(torch.load(config['model_path'], map_location=config['device']))
    model.to(config['device'])
    model.eval()
    return model

def safe_image_loader(image_path):
    """Robust image loader that handles various exceptions"""
    try:
        # First try normal loading
        with Image.open(image_path) as img:
            img.load()  # Force loading the image data now
            if img.mode == 'P':
                img = img.convert('RGBA').convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except (IOError, OSError, AttributeError) as e:
        print(f"Standard loading failed for {os.path.basename(image_path)}: {str(e)}")
        try:
            # If normal loading fails, try more aggressive approach
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img.load()
                if img.mode == 'P':
                    img = img.convert('RGBA').convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
        except Exception as e:
            print(f"Failed to recover {os.path.basename(image_path)}: {str(e)}")
            return None

def classify_images(folder_path, model):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return []
    
    image_files = sorted([f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not image_files:
        print(f"No valid images found in {folder_path}")
        return []
    
    print(f"\nClassifying {len(image_files)} images...\n")
    
    results = []
    failed_images = []
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = safe_image_loader(img_path)
        
        if img is None:
            failed_images.append(img_file)
            continue
            
        try:
            img_tensor = transform(img).unsqueeze(0).to(config['device'])
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)
            
            results.append({
                'file': img_file,
                'class': config['class_names'][pred.item()],
                'confidence': f"{probs[0][pred.item()].item():.2%}"
            })
            
            print(f"{img_file}: {results[-1]['class']} ({results[-1]['confidence']})")
            
        except Exception as e:
            print(f"Processing failed for {img_file}: {str(e)}")
            failed_images.append(img_file)
    
    # Print summary
    print("\nClassification Summary:")
    print(f"Successfully processed: {len(results)} images")
    print(f"Failed to process: {len(failed_images)} images")
    
    if results:
        flowchart_count = sum(1 for r in results if r['class'] == 'flowchart')
        print(f"\nFlowcharts detected: {flowchart_count} ({flowchart_count/len(results):.1%})")
        print(f"Non-flowcharts detected: {len(results)-flowchart_count}")
    
    if failed_images:
        print("\nFailed images:")
        for img in failed_images:
            print(f"- {img}")
    
    return results, failed_images

if __name__ == '__main__':
    print("Flowchart Classifier")
    print("=" * 40)
    
    try:
        model = load_model()
        print("Model loaded successfully\n")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    results, failed = classify_images(config['images_folder'], model)