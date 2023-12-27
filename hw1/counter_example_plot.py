import matplotlib.pyplot as plt

# Define the points
points = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
colors = ['red', 'red', 'green', 'green']
numbers = ['2', '-2', '0', '0']  # Numbers corresponding to each point

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Show integers on the axes
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1))

# Hide floats lines
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())

# Make the central axis lines thicker and darker
ax.axhline(0, color='black', linewidth=1.5)  # Horizontal line
ax.axvline(0, color='black', linewidth=1.5)  # Vertical line

# Plot the points
for point, color, number in zip(points, colors, numbers):
    x, y = point
    ax.scatter(x, y, color=color, zorder=2, s=50)
    ax.text(x + 0.1, y + 0.1, number, color='black', fontsize=10)  # Add number beside the point


# Set aspect ratio to 'equal'
ax.set_aspect('equal')

# Set labels for x-axis and y-axis
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')

# Show the plot
plt.grid(True)  # Show grid
plt.show()
