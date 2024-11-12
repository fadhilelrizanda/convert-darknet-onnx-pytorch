import sys
from tool.darknet2onnx import *

def main(cfg_file, namesfile, weight_file, batch_size):
    if batch_size <= 0:
        # Transform to ONNX with dynamic batch size
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to ONNX with specified batch size
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    
    print(f"ONNX model saved at: {onnx_path_demo}")

if __name__ == "__main__":
    print("Converting to ONNX")
    if len(sys.argv) == 5:
        cfg_file = sys.argv[1]
        namesfile = sys.argv[2]
        weight_file = sys.argv[3]
        batch_size = int(sys.argv[4])
        main(cfg_file, namesfile, weight_file, batch_size)
    else:
        print('Please run this way:\n')
        print('  python convert_darknet.py <cfgFile> <namesFile> <weightFile> <batchSize>')