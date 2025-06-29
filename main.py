import numpy as np
from uaibot import Utils

num_points = 1_000_000

# Gera pontos aleatórios uniformes entre -1 e 1 para x,y,z
cloud = np.random.uniform(low=-1.0, high=1.0, size=(num_points, 3)).astype(np.float32)

# Transformar para lista de np.array 3D se precisar (pybind11 aceita direto array numpy)
cloud_list = [cloud[i] for i in range(num_points)]

# Exemplo de origem e direção
origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
angle_deg = 15.0

filtered_points = Utils.raycast_pointcloud(cloud, origin, direction, angle_deg)

print("Pontos filtrados dentro do cone:")

print(len(filtered_points))