from google import genai
from google.genai import types
import fire


img_paths = [
    'data/poses/Panda/Panda_pose_000000_j0-090_j1-090_j2-090_j3-090_j4-090_j5-090.png',
    'data/poses/Panda/Panda_pose_000010_j0-090_j1-090_j2-090_j3+000_j4-090_j5+000.png',
    'data/poses/Panda/Panda_pose_000116_j0-090_j1+000_j2+000_j3-090_j4+090_j5+090.png',
]


def _get_media_resolution(resolution_str: str) -> types.MediaResolution:
    """
    Convert resolution string to MediaResolution enum.
    
    Args:
        resolution_str: One of 'unspecified', 'low', 'medium', 'high'
    
    Returns:
        MediaResolution enum value
    """
    resolution_map = {
        'unspecified': types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
        'low': types.MediaResolution.MEDIA_RESOLUTION_LOW,
        'medium': types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        'high': types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }
    
    resolution_lower = resolution_str.lower()
    if resolution_lower not in resolution_map:
        raise ValueError(
            f"Invalid resolution: {resolution_str}. "
            f"Must be one of: {list(resolution_map.keys())}"
        )
    
    return resolution_map[resolution_lower]


def main(
    model_name: str = 'gemini-3-flash-preview',
    media_resolution: str = 'medium',
):
    text_prompt = "Caption this robot pose with one or two sentences. Don't focus on the background or environment, just describe the robot pose."
    """
    Generate captions for robot pose images using Gemini API.
    
    Args:
        model_name: Gemini model name (e.g., 'gemini-2.5-flash', 'gemini-3-pro-preview')
        media_resolution: Media resolution setting. Options:
            - 'unspecified': Default setting (varies by model)
            - 'low': Lower token count, faster processing, lower cost
            - 'medium': Balance between detail, cost, and latency
            - 'high': Higher token count, more detail (recommended for most cases)
    
    Examples:
        # Use high resolution (recommended)
        python gemini.py --media-resolution high
        
        # Use low resolution for faster/cheaper processing
        python gemini.py --media-resolution low
        
        # Use medium resolution
        python gemini.py --media-resolution medium
    """
    client = genai.Client()
    
    # Get media resolution enum
    resolution_enum = _get_media_resolution(media_resolution)
    
    # Set global configuration
    config = types.GenerateContentConfig(
        media_resolution=resolution_enum
    )
    
    print(f"Using media resolution: {media_resolution}")
    print(f"Model: {model_name}\n")
    
    for img_path in img_paths:
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        
        # Prepare image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=[text_prompt, image_part],
            config=config
        )
        print(f"[{img_path}] {response.text}\n")


if __name__ == '__main__':
    fire.Fire(main)