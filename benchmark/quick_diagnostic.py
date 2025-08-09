import time
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.debug import visualize_array_sharding
import psutil

# ==============================
# 环境信息检测
# ==============================
def env_info():
    print("===== JAX Environment Info =====")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"Platform: {jax.default_backend()}")
    print(f"Host RAM: {psutil.virtual_memory().percent}% used")

# ==============================
# 可视化 Sharding
# ==============================
def test_sharding_visualization():
    print("\n===== Sharding Visualization Test =====")
    devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices, ('p',))
    sharding = NamedSharding(mesh, PartitionSpec('p', None))

    key = random.PRNGKey(0)
    arr = random.normal(key, (jax.local_device_count() * 4, 1024))
    arr = jax.device_put(arr, sharding)

    visualize_array_sharding(arr)

# ==============================
# 真实分片矩阵乘 Scaling 测试
# ==============================
def test_true_sharded_matmul(size=8192):
    print("\n===== True Sharded Matmul Scaling =====")
    devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices, ('p',))

    key_a, key_b = random.split(random.PRNGKey(0))
    # 按行切分 A
    a = random.normal(key_a, (jax.local_device_count() * (size // jax.local_device_count()), size))
    # B 复制到所有卡
    b = random.normal(key_b, (size, size))

    a = jax.device_put(a, NamedSharding(mesh, PartitionSpec('p', None)))
    b = jax.device_put(b, NamedSharding(mesh, PartitionSpec(None, None)))

    @jax.jit
    def sharded_matmul(x, y):
        return x @ y

    # 预热
    sharded_matmul(a, b).block_until_ready()

    # 测试
    t0 = time.time()
    sharded_matmul(a, b).block_until_ready()
    elapsed = time.time() - t0
    print(f"Sharded matmul took {elapsed:.4f} s for size {size}")

# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":
    env_info()
    test_sharding_visualization()
    test_true_sharded_matmul()
