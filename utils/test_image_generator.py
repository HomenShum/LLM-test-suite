"""
Generate sample test images for Test 6 visual LLM testing.

Creates synthetic images with various artifacts for testing:
- VR avatar artifacts (red lines, distortions)
- General test images (objects, scenes)
"""

from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
from typing import List, Tuple


def create_vr_avatar_test_image(
    image_path: str,
    avatar_id: str,
    artifacts: List[str] = None,
    size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Create a synthetic VR avatar test image with specified artifacts.
    
    Args:
        image_path: Path to save the image
        avatar_id: Avatar identifier
        artifacts: List of artifacts to include
        size: Image size (width, height)
    """
    # Create base image
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw avatar silhouette (simple humanoid shape)
    # Head
    head_center = (size[0] // 2, size[1] // 4)
    head_radius = 50
    draw.ellipse(
        [head_center[0] - head_radius, head_center[1] - head_radius,
         head_center[0] + head_radius, head_center[1] + head_radius],
        fill=(200, 180, 160), outline=(100, 100, 100), width=2
    )
    
    # Body
    body_top = head_center[1] + head_radius
    body_bottom = size[1] - 100
    body_width = 80
    draw.rectangle(
        [size[0] // 2 - body_width // 2, body_top,
         size[0] // 2 + body_width // 2, body_bottom],
        fill=(150, 150, 200), outline=(100, 100, 100), width=2
    )
    
    # Arms
    arm_y = body_top + 20
    draw.line(
        [size[0] // 2 - body_width // 2, arm_y,
         size[0] // 2 - body_width // 2 - 60, arm_y + 80],
        fill=(150, 150, 200), width=15
    )
    draw.line(
        [size[0] // 2 + body_width // 2, arm_y,
         size[0] // 2 + body_width // 2 + 60, arm_y + 80],
        fill=(150, 150, 200), width=15
    )
    
    # Legs
    leg_y = body_bottom
    draw.line(
        [size[0] // 2 - 20, leg_y,
         size[0] // 2 - 20, size[1] - 10],
        fill=(100, 100, 150), width=20
    )
    draw.line(
        [size[0] // 2 + 20, leg_y,
         size[0] // 2 + 20, size[1] - 10],
        fill=(100, 100, 150), width=20
    )
    
    # Add artifacts if specified
    if artifacts:
        if "red lines in eyes" in artifacts:
            # Draw red lines in eye area
            eye_y = head_center[1] - 10
            draw.line([head_center[0] - 20, eye_y, head_center[0] - 10, eye_y], fill=(255, 0, 0), width=3)
            draw.line([head_center[0] + 10, eye_y, head_center[0] + 20, eye_y], fill=(255, 0, 0), width=3)
        
        if "finger movement issues" in artifacts:
            # Draw distorted fingers
            for i in range(5):
                x = size[0] // 2 + body_width // 2 + 60 + i * 8
                y = arm_y + 80 + random.randint(-10, 10)
                draw.line([x, y, x, y + 15], fill=(255, 100, 100), width=2)
        
        if "avatar distortions" in artifacts:
            # Add wavy distortion lines
            for i in range(5):
                y = body_top + i * 30
                points = [(size[0] // 2 - body_width // 2 + j * 10, y + random.randint(-5, 5)) 
                         for j in range(body_width // 10)]
                draw.line(points, fill=(255, 255, 0), width=2)
        
        if "clothing distortions" in artifacts:
            # Add texture artifacts on clothing
            for _ in range(20):
                x = random.randint(size[0] // 2 - body_width // 2, size[0] // 2 + body_width // 2)
                y = random.randint(body_top, body_bottom)
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 0, 255))
    
    # Add label
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), f"Avatar: {avatar_id}", fill=(0, 0, 0), font=font)
    
    if artifacts:
        artifact_text = ", ".join(artifacts[:2])  # Show first 2 artifacts
        draw.text((10, size[1] - 30), f"Artifacts: {artifact_text}", fill=(255, 0, 0), font=font)
    
    # Save image
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)


def create_general_test_image(
    image_path: str,
    scene_type: str = "objects",
    size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Create a general test image for Mode B testing.
    
    Args:
        image_path: Path to save the image
        scene_type: Type of scene ("objects", "landscape", "abstract")
        size: Image size (width, height)
    """
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if scene_type == "objects":
        # Draw simple objects
        # Circle
        draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
        # Square
        draw.rectangle([200, 50, 300, 150], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
        # Triangle
        draw.polygon([(350, 150), (400, 50), (450, 150)], fill=(0, 0, 255), outline=(0, 0, 0), width=2)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text((150, 250), "Test Objects", fill=(0, 0, 0), font=font)
    
    elif scene_type == "landscape":
        # Sky
        draw.rectangle([0, 0, size[0], size[1] // 2], fill=(135, 206, 235))
        # Ground
        draw.rectangle([0, size[1] // 2, size[0], size[1]], fill=(34, 139, 34))
        # Sun
        draw.ellipse([400, 50, 480, 130], fill=(255, 255, 0), outline=(255, 200, 0), width=3)
        # Tree
        draw.rectangle([100, 200, 130, 350], fill=(139, 69, 19))
        draw.ellipse([60, 150, 170, 250], fill=(0, 128, 0))
    
    elif scene_type == "abstract":
        # Random shapes and colors
        for _ in range(20):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            shape = random.choice(['ellipse', 'rectangle', 'line'])
            if shape == 'ellipse':
                draw.ellipse([x1, y1, x2, y2], fill=color)
            elif shape == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.line([x1, y1, x2, y2], fill=color, width=5)
    
    # Save image
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)


def generate_sample_dataset():
    """Generate a complete sample dataset for testing."""
    base_dir = Path("test_dataset/visual_llm_images")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating sample test images...")
    
    # VR Avatar test images (Mode A)
    vr_avatars = [
        ("avatar_001", ["red lines in eyes"]),
        ("avatar_002", ["finger movement issues"]),
        ("avatar_003", ["avatar distortions"]),
        ("avatar_004", ["clothing distortions"]),
        ("avatar_005", ["red lines in eyes", "finger movement issues"]),
        ("avatar_006", []),  # No artifacts
    ]
    
    for avatar_id, artifacts in vr_avatars:
        image_path = base_dir / f"{avatar_id}.png"
        create_vr_avatar_test_image(str(image_path), avatar_id, artifacts)
        print(f"  ✓ Created {image_path}")
    
    # General test images (Mode B)
    general_images = [
        ("test_objects.png", "objects"),
        ("test_landscape.png", "landscape"),
        ("test_abstract.png", "abstract"),
    ]
    
    for filename, scene_type in general_images:
        image_path = base_dir / filename
        create_general_test_image(str(image_path), scene_type)
        print(f"  ✓ Created {image_path}")
    
    print(f"\n✅ Generated {len(vr_avatars) + len(general_images)} test images in {base_dir}")
    return base_dir


if __name__ == "__main__":
    generate_sample_dataset()

