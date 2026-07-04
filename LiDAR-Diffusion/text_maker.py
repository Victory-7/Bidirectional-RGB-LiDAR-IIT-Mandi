import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import os
import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class KITTIImageDescriber:
    def __init__(self, device=None):
        """
        Initialize the image describer with CLIP and BLIP models
        """
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model using Hugging Face transformers with safetensors
        print("Loading CLIP model...")
        try:
            # Try loading with safetensors first
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float32,
                use_safetensors=True
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            print("Attempting to load with different settings...")
            # Fallback: try loading without safetensors
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float32,
                use_safetensors=False,
                from_tf=False
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP model loaded with fallback settings!")
        
        # Load BLIP model for detailed captioning
        print("Loading BLIP model...")
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float32
            ).to(self.device)
            print("✅ BLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading BLIP: {e}")
            print("BLIP model failed to load, but CLIP will still work.")
            self.blip_model = None
            self.blip_processor = None
        
        # KITTI-specific categories
        self.categories = [
            "car", "sedan", "suv", "truck", "van", "bus", "motorcycle", "bicycle", 
            "pedestrian", "person", "cyclist", "road", "highway", "street", 
            "building", "house", "tree", "sky", "clouds", "traffic sign", 
            "traffic light", "street light", "sidewalk", "crosswalk", 
            "parking lot", "intersection", "vehicle", "urban scene", 
            "driving scene", "outdoor", "city", "daytime", "nighttime",
            "rainy", "sunny", "cloudy", "foggy", "clear sky"
        ]
        
        # Create output directories
        self.output_dir = "kitti_descriptions"
        self.text_dir = os.path.join(self.output_dir, "text_descriptions")
        self.json_dir = os.path.join(self.output_dir, "json_results")
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        print(f"📁 Output folders created:")
        print(f"   - Text files: {self.text_dir}")
        print(f"   - JSON files: {self.json_dir}")
        print(f"   - Visualizations: {self.visualizations_dir}")
        print("✅ All models loaded successfully!")
    
    def load_image(self, image_path):
        """
        Load image from path
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def classify_image(self, image):
        """
        Classify image into KITTI-specific categories using CLIP
        """
        if image is None:
            return []
        
        try:
            # Prepare inputs
            inputs = self.clip_processor(
                text=[f"a photo of a {cat}" for cat in self.categories], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top categories
            values, indices = probs[0].topk(10)
            top_categories = []
            for value, index in zip(values, indices):
                top_categories.append((self.categories[index], value.item()))
            
            return top_categories
        except Exception as e:
            print(f"Error in classification: {e}")
            return []
    
    def generate_detailed_caption(self, image):
        """
        Generate detailed caption using BLIP
        """
        if image is None:
            return "No image available"
        
        if self.blip_model is None or self.blip_processor is None:
            return "BLIP model not available"
        
        try:
            # Process image for BLIP
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.9
                )
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Unable to generate caption"
    
    def generate_kitti_specific_description(self, image, categories):
        """
        Generate KITTI-specific description
        """
        # Extract scene information from categories
        scene_elements = []
        for cat, score in categories[:5]:
            if score > 0.1:  # Only include if confidence is reasonable
                scene_elements.append(cat)
        
        # Get basic caption
        caption = self.generate_detailed_caption(image)
        
        # Construct KITTI-specific description
        if not scene_elements:
            return caption
        
        description = f"KITTI driving scene with {', '.join(scene_elements)}. "
        
        # Add specific details based on categories
        if any(word in scene_elements for word in ["car", "truck", "bus", "vehicle"]):
            description += "Multiple vehicles are visible on the road. "
        if "pedestrian" in scene_elements or "person" in scene_elements:
            description += "Pedestrians are present in the scene. "
        if "traffic sign" in scene_elements:
            description += "Traffic signs are visible. "
        if "traffic light" in scene_elements:
            description += "Traffic lights are present. "
        if "building" in scene_elements:
            description += "Urban buildings flank the street. "
        if "tree" in scene_elements:
            description += "Trees line the roadway. "
        if "sky" in scene_elements:
            description += "Sky is visible. "
        
        # Add weather/lighting conditions
        if any(word in scene_elements for word in ["sunny", "clear sky"]):
            description += "Weather appears to be clear and sunny. "
        elif "cloudy" in scene_elements:
            description += "Weather appears to be cloudy. "
        elif "rainy" in scene_elements:
            description += "Weather appears to be rainy. "
        
        # Add time of day
        if "daytime" in scene_elements:
            description += "Scene is captured during daytime. "
        elif "nighttime" in scene_elements:
            description += "Scene is captured at nighttime. "
        
        description += f"Overall scene: {caption}"
        
        return description
    
    def save_as_txt(self, result):
        """
        Save the description as a TXT file
        """
        image_name = result['image_name']
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(self.text_dir, txt_filename)
        
        # Create formatted text content
        content = []
        content.append("=" * 70)
        content.append("KITTI IMAGE DESCRIPTION")
        content.append("=" * 70)
        content.append(f"Image: {result['image_name']}")
        content.append(f"Image Size: {result['image_size']}")
        content.append(f"Processed: {result['timestamp']}")
        content.append("")
        content.append("-" * 70)
        content.append("DETAILED DESCRIPTION")
        content.append("-" * 70)
        content.append(result['detailed_description'])
        content.append("")
        content.append("-" * 70)
        content.append("BASIC CAPTION")
        content.append("-" * 70)
        content.append(result['basic_caption'])
        content.append("")
        content.append("-" * 70)
        content.append("TOP CATEGORIES WITH CONFIDENCE SCORES")
        content.append("-" * 70)
        
        for cat, score in result['categories'][:10]:
            bar_length = int(score * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            content.append(f"{cat:25s} : {score:6.2%} {bar}")
        
        content.append("")
        content.append("-" * 70)
        content.append("ALL CATEGORIES (with confidence)")
        content.append("-" * 70)
        
        for cat, score in result['categories']:
            content.append(f"{cat:25s} : {score:6.2%}")
        
        content.append("")
        content.append("=" * 70)
        content.append("Generated by CLIP-based KITTI Image Describer")
        content.append("=" * 70)
        
        # Write to file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return txt_path
    
    def process_single_image(self, image_path):
        """
        Process a single image and return detailed description
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Get classification
        categories = self.classify_image(image)
        
        # Get KITTI-specific description
        description = self.generate_kitti_specific_description(image, categories)
        
        # Get basic caption
        basic_caption = self.generate_detailed_caption(image)
        
        # Get image size info
        width, height = image.size
        
        result = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "image_size": f"{width}x{height}",
            "basic_caption": basic_caption,
            "detailed_description": description,
            "categories": categories,
            "top_categories": [cat for cat, score in categories[:5]],
            "confidence_scores": {cat: score for cat, score in categories[:5]},
            "timestamp": datetime.now().isoformat()
        }
        
        # Save as TXT immediately
        txt_path = self.save_as_txt(result)
        result['txt_path'] = txt_path
        
        print(f"  ✅ Saved: {txt_path}")
        return result
    
    def process_directory(self, image_dir, max_images=None):
        """
        Process all images in a directory
        """
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        # Sort for consistent processing
        image_files.sort()
        
        # Limit number of images if specified
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"Limiting to first {max_images} images")
        
        print(f"\n📸 Found {len(image_files)} images to process")
        print("=" * 70)
        
        results = []
        successful_txt = 0
        
        # Process each image with progress bar
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            result = self.process_single_image(image_path)
            if result:
                results.append(result)
                successful_txt += 1
            
            # Save intermediate JSON results every 10 images
            if (i + 1) % 10 == 0:
                self.save_json_results(results, f"kitti_results_partial_{i+1}.json")
        
        print(f"\n✅ Successfully created {successful_txt} TXT description files")
        return results
    
    def save_json_results(self, results, filename=None):
        """
        Save results to JSON file
        """
        if filename is None:
            filename = f"kitti_descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.json_dir, filename)
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            serializable_result["categories"] = [(cat, score) for cat, score in result["categories"]]
            serializable_result["confidence_scores"] = result["confidence_scores"]
            serializable_result["top_categories"] = result["top_categories"]
            # Remove path info that might not be serializable
            serializable_result.pop("txt_path", None)
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 JSON results saved to {filepath}")
        return filepath
    
    def generate_summary_report(self, results):
        """
        Generate a summary report of all processed images
        """
        if not results:
            print("No results to summarize")
            return
        
        # Count categories
        category_counts = {}
        for result in results:
            for cat, score in result["categories"][:5]:
                if cat in category_counts:
                    category_counts[cat] += 1
                else:
                    category_counts[cat] = 1
        
        # Sort by frequency
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate report
        report = {
            "total_images": len(results),
            "most_common_objects": sorted_categories[:10],
            "image_processing_date": datetime.now().isoformat(),
            "average_confidence_per_category": {}
        }
        
        # Calculate average confidence for top categories
        category_confidences = {}
        for result in results:
            for cat, score in result["categories"][:5]:
                if cat not in category_confidences:
                    category_confidences[cat] = []
                category_confidences[cat].append(score)
        
        for cat, scores in category_confidences.items():
            report["average_confidence_per_category"][cat] = np.mean(scores)
        
        # Save report as JSON
        report_path = os.path.join(self.json_dir, f"kitti_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save summary as TXT
        summary_txt_path = os.path.join(self.text_dir, "SUMMARY_REPORT.txt")
        with open(summary_txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("KITTI IMAGE DESCRIPTION SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Top 10 Most Common Objects:\n")
            f.write("-" * 50 + "\n")
            for i, (cat, count) in enumerate(sorted_categories[:10], 1):
                percentage = (count / len(results)) * 100
                bar = "█" * int(percentage / 2)
                f.write(f"{i:2d}. {cat:25s} : {count:4d} images ({percentage:5.1f}%) {bar}\n")
            
            f.write("\nAverage Confidence Scores:\n")
            f.write("-" * 50 + "\n")
            for cat, avg_conf in sorted(report["average_confidence_per_category"].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{cat:25s} : {avg_conf:6.2%}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"All TXT files saved in: {self.text_dir}\n")
            f.write(f"All JSON files saved in: {self.json_dir}\n")
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 KITTI IMAGE DESCRIPTION SUMMARY")
        print("=" * 70)
        print(f"Total images processed: {len(results)}")
        print(f"\n📁 TXT files saved in: {self.text_dir}")
        print(f"📁 JSON files saved in: {self.json_dir}")
        print("\nTop 10 Most Common Objects:")
        for i, (cat, count) in enumerate(sorted_categories[:10], 1):
            percentage = (count / len(results)) * 100
            print(f"  {i:2d}. {cat:25s} : {count:4d} images ({percentage:5.1f}%)")
        print(f"\n✅ Summary report saved to: {summary_txt_path}")
        
        return report
    
    def display_image_with_description(self, image_path):
        """
        Display image with its description
        """
        result = self.process_single_image(image_path)
        if result is None:
            print("Failed to process image")
            return
        
        image = self.load_image(image_path)
        
        # Display image with matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(f"KITTI Image: {result['image_name']}", fontsize=12)
        ax1.axis('off')
        
        # Display description
        ax2.axis('off')
        ax2.text(0.1, 0.95, "📝 KITTI Description:", fontsize=14, fontweight='bold')
        
        # Wrap description text
        desc_text = result['detailed_description']
        wrapped_desc = '\n'.join([desc_text[i:i+60] for i in range(0, len(desc_text), 60)])
        ax2.text(0.1, 0.80, wrapped_desc, fontsize=10, wrap=True)
        
        ax2.text(0.1, 0.60, "🏷️ Top Categories:", fontsize=13, fontweight='bold')
        y_pos = 0.55
        for cat, score in result['categories'][:5]:
            bar_length = score * 100
            ax2.text(0.1, y_pos, f"• {cat}:", fontsize=11, fontweight='bold')
            ax2.barh(y_pos-0.01, bar_length, left=0.25, height=0.02, color='blue', alpha=0.6)
            ax2.text(0.25 + bar_length/2, y_pos, f"{score:.1%}", fontsize=10, va='center')
            y_pos -= 0.08
        
        # Save visualization
        vis_path = os.path.join(self.visualizations_dir, f"{os.path.splitext(result['image_name'])[0]}_vis.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {vis_path}")
        
        plt.tight_layout()
        plt.show()

# Main execution for KITTI dataset
def process_kitti_dataset():
    """
    Main function to process KITTI dataset
    """
    # Path to your KITTI images
    kitti_path = "/home/teaching/Suhani/project/kitti_object/training/image_2"
    
    # Check if path exists
    if not os.path.exists(kitti_path):
        print(f"❌ Error: Path {kitti_path} does not exist!")
        return
    
    print(f"📁 Processing KITTI images from: {kitti_path}")
    print("=" * 70)
    
    # Initialize describer
    describer = KITTIImageDescriber()
    
    # Process all images
    results = describer.process_directory(kitti_path, max_images=None)  # Process all images
    
    if results:
        # Save all results as JSON
        describer.save_json_results(results)
        
        # Generate summary report
        describer.generate_summary_report(results)
        
        print(f"\n✅ Processing complete! Processed {len(results)} images")
        print(f"📁 All results saved in: {describer.output_dir}")
    else:
        print("❌ No images were processed successfully!")

# Function to process only a few images for testing
def test_with_few_images():
    """
    Test the pipeline with just a few images
    """
    kitti_path = "/home/teaching/Suhani/project/kitti_object/training/image_2"
    
    if not os.path.exists(kitti_path):
        print(f"❌ Error: Path {kitti_path} does not exist!")
        return
    
    print("🧪 Testing with first 5 images...")
    print("=" * 70)
    
    describer = KITTIImageDescriber()
    
    # Get first 5 images
    image_files = glob.glob(os.path.join(kitti_path, "*.png"))[:5]
    
    if not image_files:
        image_files = glob.glob(os.path.join(kitti_path, "*.jpg"))[:5]
    
    if not image_files:
        print("No images found in the directory!")
        return
    
    print(f"\n📸 Found {len(image_files)} test images")
    
    for image_path in image_files:
        print(f"\n{'='*50}")
        result = describer.process_single_image(image_path)
        if result:
            print(f"✅ TXT saved: {result['txt_path']}")
            print(f"📝 Description: {result['detailed_description'][:150]}...")
            print(f"🏷️ Categories: {', '.join(result['top_categories'])}")
        else:
            print(f"❌ Failed to process: {os.path.basename(image_path)}")
    
    # Display one image with description
    if image_files:
        print("\n📊 Displaying first image with description...")
        describer.display_image_with_description(image_files[0])

# Install required packages function
def install_requirements():
    """
    Install required packages
    """
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "opencv-python",
        "matplotlib",
        "tqdm",
        "numpy",
        "requests",
        "safetensors"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        os.system(f"pip install {package}")
    print("✅ Packages installed!")

# Fix for PyTorch 2.6 compatibility
def fix_torch_compatibility():
    """
    Fix torch compatibility issues
    """
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # For PyTorch 2.6+, we need to set some environment variables
    if torch.__version__ >= "2.6.0":
        os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
        print("Applied PyTorch 2.6+ compatibility settings")

if __name__ == "__main__":
    print("🚗 KITTI Dataset Image Describer with CLIP")
    print("=" * 70)
    
    # Fix torch compatibility
    fix_torch_compatibility()
    
    # Check and install required packages if needed
    try:
        import transformers
        import torch
    except ImportError:
        print("⚠️  Some packages are missing. Installing...")
        install_requirements()
    
    # Choose mode
    print("\nSelect mode:")
    print("1: Full processing (process all images)")
    print("2: Test with few images (first 5 images)")
    
    mode = input("\nEnter choice (1 or 2): ").strip()
    
    if mode == "1":
        process_kitti_dataset()
    elif mode == "2":
        test_with_few_images()
    else:
        print("Invalid mode. Running test mode...")
        test_with_few_images()
    
    print("\n✨ Done!")