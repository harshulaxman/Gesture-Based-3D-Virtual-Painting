from vpython import sphere, vector, color, rate

# Create a 3D object
ball = sphere(pos=vector(0, 0, 0), radius=0.5, color=color.cyan)

# Keep window open
while True:
    rate(30)  # keeps the loop running at 30fps
