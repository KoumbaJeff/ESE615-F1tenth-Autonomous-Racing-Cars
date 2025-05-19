# import numpy as np
# import csv
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev

# # Define key points from RViz (Levine loop corners)
# waypoints = np.array([
#     [-0.6799850463867188, 0.1838083267211914], 
#     [1.36, -0.12],   
#     [5.43, -0.24],
#     [8.087255477905273, -0.6382672786712646],  
#     [8.7, 5.19], 
#     [9.38, 10.19],
#     [10.42, 17.19],
#     [10.985471725463867, 22.652149200439453],  
#     [7.76, 23.34],
#     [4.5, 23.8],
#     [2.19050931930542, 23.707059860229492],  
#     [1.5, 18.92], 
#     [0.45, 10],
#     [-0.23, 5.12],
#     [-0.6799850463867188, 0.1838083267211914]  # Closing the loop
# ])

# # Extract x and y
# x, y = waypoints[:, 0], waypoints[:, 1]

# # Clamped B-spline with minimal corner smoothing
# tck, u = splprep([x, y], s=0.5, k=2)  # k=2 for sharper corners
# u_fine = np.linspace(0, 1, 100)  # Generate 100 points
# x_smooth, y_smooth = splev(u_fine, tck)

# # Save to CSV
# csv_filename = "levine.csv"
# with open(csv_filename, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["x", "y"])
#     for xi, yi in zip(x_smooth, y_smooth):
#         writer.writerow([xi, yi])

# print(f"Waypoints saved to {csv_filename}")

# # Plot
# plt.plot(x, y, 'ro-', label="Original Corners")  # Red dots: Key points
# plt.plot(x_smooth, y_smooth, 'g-', label="Clamped B-Spline (k=2)")  # Green: Smoothed Path
# plt.legend()
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Clamped B-Spline for Levine Loop")
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import csv  # Import CSV module for saving data

def generate_smooth_path(waypoints, num_points=100):
    waypoints = np.array(waypoints)
    x, y = waypoints[:, 0], waypoints[:, 1]
    
    smoothed_path = []
    for i in range(0, len(waypoints), 3):  # Process each corner set
        if i == 0:
            smoothed_path.append(waypoints[i])  # Start point
        else:
            # Add straight-line segment before the corner
            straight_segment = np.linspace(smoothed_path[-1], waypoints[i], num=10)
            smoothed_path.extend(straight_segment)
        
        # Extract corner points
        corner_points = waypoints[i:i+3]
        if len(corner_points) < 3:
            break
        
        # Use splprep to create a smooth curve through the corner
        tck, u = splprep([corner_points[:, 0], corner_points[:, 1]], s=0, k=2)
        u_fine = np.linspace(0, 1, num_points // len(waypoints))
        smooth_x, smooth_y = splev(u_fine, tck)
        smoothed_path.extend(np.column_stack((smooth_x, smooth_y)))
    
    # Add the final straight segment to loop back to the start
    final_segment = np.linspace(smoothed_path[-1], waypoints[0], num=10)
    smoothed_path.extend(final_segment)
    
    return np.array(smoothed_path)

# Define waypoints with closer neighboring points per corner (entry, corner, exit)
waypoints = np.array([
    [6.5, -0.3808], [8.087255, -0.638267], [8.5, 0.55], # First corner
    [10.3, 20.5], [10.985471, 22.652149], [8.0, 22.9],  # Second corner
    [3.5, 23.5], [2.190509, 23.707059], [1.5, 22.5],  # Third corner
    [0.0, 2.0], [-0.679985, 0.183808], [1.5, 0.0]   # Fourth corner (loop back)
])

# Generate the smooth path with straights and corners
smooth_path = generate_smooth_path(waypoints)

# Save to CSV
csv_filename = "levine.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])
    for xi, yi in smooth_path:
        writer.writerow([xi, yi])

print(f"Waypoints saved to {csv_filename}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', label="Waypoints (Corners)")
plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'b-', label="Smooth Path")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Smooth Path with Straights and Rounded Corners")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

