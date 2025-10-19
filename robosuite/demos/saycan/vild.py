"""
This demo implements ViLD (Vision-Language Detection) for robosuite.
"""

import numpy as np
import torch
import clip
from PIL import Image
import cv2

class ViLDDetector:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define object categories
        self.categories = [
            "blue block", "red block", "green block", "yellow block",
            "blue bowl", "red bowl", "green bowl", "yellow bowl"
        ]
        
        # Create text embeddings
        self.text_descriptions = [f"a photo of a {cat}" for cat in self.categories]
        self.text_tokens = clip.tokenize(self.text_descriptions).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
    def detect_objects(self, image):
        """Detect objects in image using CLIP zero-shot classification."""
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            similarity = similarity[0].cpu().numpy()
        
        # Get detected objects above threshold
        threshold = 0.2
        detected = []
        for score, category in zip(similarity, self.categories):
            if score > threshold:
                detected.append({
                    'category': category,
                    'score': float(score)
                })
                
        return detected
        
    def get_scene_description(self, image):
        """Generate natural language scene description."""
        detected = self.detect_objects(image)
        
        # Format description
        if not detected:
            return "No objects detected in the scene."
            
        desc = "Scene contains: "
        obj_strings = []
        for obj in detected:
            obj_strings.append(f"{obj['category']} ({obj['score']:.2f})")
        desc += ", ".join(obj_strings)
        
        return desc

def main():
    import robosuite as suite
    
    # Create environment
    env = suite.make(
        "Stack",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    
    # Initialize detector
    detector = ViLDDetector()
    
    # Reset environment and get image
    obs = env.reset()
    env.viewer.set_camera(camera_id=0)
    image = env.get_camera_image()
    
    # Get scene description
    desc = detector.get_scene_description(image)
    print(desc)
    
if __name__ == "__main__":
    main()