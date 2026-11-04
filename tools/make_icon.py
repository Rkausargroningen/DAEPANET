"""Generate assets/app.ico and assets/app.png for Water Network Modeler."""

import math
from pathlib import Path
from PIL import Image, ImageDraw

def create_app_icon(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render at high resolution (1024x1024) for supersampled antialiasing
    scale = 4
    size = 256 * scale  # 1024x1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Dark-blue rounded square background
    # Colors: Dark slate/navy blue #0F172A
    dark_blue = (15, 23, 42, 255)
    margin = 16 * scale
    radius = 50 * scale
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=dark_blue
    )

    # Subtle inner border for depth
    border_color = (30, 41, 59, 255)
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        outline=border_color,
        width=int(2 * scale)
    )

    # 2. Blue water drop
    # Vibrant water blue #0284C7 / #38BDF8
    water_blue = (14, 165, 233, 255)  # Sky blue 500

    cx = size // 2
    top_y = 60 * scale
    bulb_cy = 155 * scale
    bulb_r = 60 * scale

    dy = bulb_cy - top_y
    # Tangent calculation
    # sin(theta) = bulb_r / dy
    theta = math.asin(bulb_r / dy)
    contact_y = bulb_cy - bulb_r * math.sin(theta)
    dx = bulb_r * math.cos(theta)
    right_contact = (cx + dx, contact_y)
    left_contact = (cx - dx, contact_y)

    # Arc angles in degrees
    # theta is angle from horizontal: -asin(sin(theta))
    # In PIL: 0 is 3 o'clock, 90 is 6 o'clock, 180 is 9 o'clock, 270 is 12 o'clock
    start_angle = -math.degrees(math.asin(math.sin(theta)))  # approx -34.8 deg
    end_angle = 180 - start_angle  # approx 214.8 deg

    # Build polygon points along circle from right_contact clockwise to left_contact
    drop_pts = [(cx, top_y)]
    steps = 120
    for i in range(steps + 1):
        a_deg = start_angle + (end_angle - start_angle) * (i / steps)
        a_rad = math.radians(a_deg)
        px = cx + bulb_r * math.cos(a_rad)
        py = bulb_cy + bulb_r * math.sin(a_rad)
        drop_pts.append((px, py))

    draw.polygon(drop_pts, fill=water_blue)

    # 3. Network graph inside water drop: 3 small white nodes connected by lines
    # Nodes layout
    n1 = (cx, int(122 * scale))            # Top node
    n2 = (int(105 * scale), int(172 * scale)) # Bottom-left node
    n3 = (int(151 * scale), int(172 * scale)) # Bottom-right node

    line_color = (255, 255, 255, 240)
    line_width = int(4 * scale)

    # Connect nodes with lines (a triangle network)
    draw.line([n1, n2], fill=line_color, width=line_width)
    draw.line([n2, n3], fill=line_color, width=line_width)
    draw.line([n3, n1], fill=line_color, width=line_width)

    # Draw white circular nodes
    node_radius = int(7 * scale)
    node_fill = (255, 255, 255, 255)
    node_outline = (14, 116, 144, 255)  # Cyan/teal outline for definition
    node_outline_w = int(1.5 * scale)

    for nx, ny in (n1, n2, n3):
        draw.ellipse(
            [nx - node_radius, ny - node_radius, nx + node_radius, ny + node_radius],
            fill=node_fill,
            outline=node_outline,
            width=node_outline_w
        )

    # Downsample using high quality Lanczos filter
    final_256 = img.resize((256, 256), Image.Resampling.LANCZOS)

    # Save app.png
    png_path = output_dir / "app.png"
    final_256.save(png_path, "PNG")
    print(f"Saved {png_path}")

    # Generate multi-size ICO
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    ico_images = [img.resize(s, Image.Resampling.LANCZOS) for s in ico_sizes]

    ico_path = output_dir / "app.ico"
    final_256.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"Saved {ico_path} with sizes {ico_sizes}")


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "assets"
    create_app_icon(assets_dir)
