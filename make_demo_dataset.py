from PIL import Image, ImageDraw
import os

# Paths
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

# Create 3 demo "portrait" images with colored backgrounds
for i, bg in enumerate([(200,200,255), (255,200,200), (200,255,200)], start=1):
    # Base image
    img = Image.new("RGB", (400,600), bg)  # 400x600 px "portrait"
    mask = Image.new("L", (400,600), 0)    # black mask
    
    # Draw a simple "person shape" (circle head + rectangle body)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    # Head
    draw_img.ellipse((150,50,250,150), fill=(255,224,189))   # skin tone
    draw_mask.ellipse((150,50,250,150), fill=255)
    
    # Body
    draw_img.rectangle((130,150,270,450), fill=(50,100,200)) # shirt
    draw_mask.rectangle((130,150,270,450), fill=255)
    
    # Save both
    img.save(f"data/images/IMG_{i:04d}.jpg")
    mask.save(f"data/masks/IMG_{i:04d}.png")

print("Demo dataset created in data/images and data/masks")