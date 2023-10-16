import tensorflow as tf

# Listar todos los dispositivos físicos disponibles
devices = tf.config.experimental.list_physical_devices()

if not devices:
    print("No se encontraron dispositivos físicos.")
else:
    print(f"Se encontraron {len(devices)} dispositivo(s) físico(s).")

# Verificar si hay GPUs disponibles
gpus = tf.config.experimental.list_physical_devices('GPU')

if not gpus:
    print("No se encontraron GPUs.")
else:
    print(f"Se encontraron {len(gpus)} GPU(s). Detalles:")
    for gpu in gpus:
        print(f"- Nombre: {gpu.name}, Tipo: {gpu.device_type}")

# Verificar si TensorFlow puede acceder a la GPU
if tf.test.is_built_with_cuda():
    print("TensorFlow fue compilado con soporte para CUDA.")
else:
    print("TensorFlow no fue compilado con soporte para CUDA.")

# Verificar la versión de CUDA
if tf.test.is_built_with_cuda():
    print(f"Versión de CUDA: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(
        f"Versión de cuDNN: {tf.sysconfig.get_build_info()['cudnn_version']}")
