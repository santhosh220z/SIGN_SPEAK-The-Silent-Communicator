import numpy as np
np.object = object
import sys
from tensorflowjs import converters
if len(sys.argv) != 3:
    print('Usage: python convert_model.py <input_model> <output_dir>')
    sys.exit(1)
input_path = sys.argv[1]
output_dir = sys.argv[2]
converters.converter.convert_keras_model(input_path, output_dir)
print('Conversion completed')
