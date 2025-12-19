# src/run_full_test.py
import os
import json
import glob
from tqdm import tqdm
from pipeline.pipeline import CaptionPipeline

def main():
    # Use absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize pipeline with GPU
    print("Initializing pipeline...")
    pipe = CaptionPipeline(
        dino_ckpt=os.path.join(BASE_DIR, "outputs/dino/epoch_12.pth"),
        vit_ckpt=os.path.join(BASE_DIR, "outputs/vit/vit_env.pth"),   
        exp_ckpt=os.path.join(BASE_DIR, "outputs/expansionnet/expnetv2_final.pth"),
        device="cuda"  # Use GPU
    )
    print("Pipeline initialized!")
    
    # Test dataset directories
    test_dirs = [
        "VS_EO_SU_DT",
        "VS_EO_SU_NT", 
        "VS_EO_WI_DT",
        "VS_IR_SU_NT"  # Re-enabled after fix
    ]
    
    # Collect all test images
    test_images = []
    for test_dir in test_dirs:
        img_dir = os.path.join(BASE_DIR, "dataset/test/image", test_dir)
        images = glob.glob(os.path.join(img_dir, "*.jpg"))
        test_images.extend(images)
    
    print(f"Found {len(test_images)} test images")
    
    # Load ground truth labels
    label_dir = os.path.join(BASE_DIR, "dataset/test/label")
    
    results = []
    
    # Run inference on all test images
    for img_path in tqdm(test_images, desc="Running inference"):
        # Get corresponding label file
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # Determine label subdirectory based on image path
        if "VS_EO_SU_DT" in img_path:
            label_subdir = "VL_EO_SU_DT"
        elif "VS_EO_SU_NT" in img_path:
            label_subdir = "VL_EO_SU_NT"
        elif "VS_EO_WI_DT" in img_path:
            label_subdir = "VL_EO_WI_DT"
        elif "VS_IR_SU_NT" in img_path:
            label_subdir = "VL_IR_SU_NT"
        else:
            label_subdir = None
        
        # Load ground truth
        gt_caption = None
        if label_subdir:
            label_path = os.path.join(label_dir, label_subdir, f"{img_name_no_ext}.json")
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    gt_caption = label_data.get('caption', '')
        
        # Run inference
        try:
            pred_caption = pipe.run(img_path, debug=False)
            
            results.append({
                'image': img_name,
                'image_path': img_path,
                'predicted': pred_caption,
                'ground_truth': gt_caption
            })
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            results.append({
                'image': img_name,
                'image_path': img_path,
                'predicted': f"ERROR: {str(e)}",
                'ground_truth': gt_caption
            })
    
    # Save results
    output_file = os.path.join(BASE_DIR, "test_results_full.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total images processed: {len(results)}")
    print(f"{'='*60}")
    
    # Print sample results
    print("\nSample results (first 5):")
    for i, result in enumerate(results[:5]):
        print(f"\n[{i+1}] {result['image']}")
        print(f"  Predicted: {result['predicted']}")
        print(f"  Ground Truth: {result['ground_truth']}")

if __name__ == "__main__":
    main()
