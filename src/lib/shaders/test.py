max_radius = 10

radius_covered = 0
step_size = 1

print("(0, 0)")
while radius_covered < max_radius:
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            
            x = dx * (radius_covered + (step_size // 2) + 1)
            y = dy * (radius_covered + (step_size // 2) + 1)
            print(f"({x}, {y})")

    radius_covered += step_size    
    step_size += 2
