
import argparse
import numpy as np
from skimage import io
from math import radians, sin, cos
import os
from question1 import nearest_interpolation, to_rgb
from question2 import bilinear_interpolation
from question3 import bicubic_interpolation
from question4 import lagrange_interpolation

def transform_image(img, scale=1.0, angle=0.0, output_size=None, method='nearest'):
    interp_func = {
        'nearest': nearest_interpolation,
        'bilinear': bilinear_interpolation,
        'bicubic': bicubic_interpolation,
        'lagrange': lagrange_interpolation
    }.get(method)

    if interp_func is None:
        raise ValueError(f"Interpolação '{method}' não é suportada.")

    img = to_rgb(img)
    h_in, w_in = img.shape[:2]
    if output_size:
        w_out, h_out = output_size
    else:
        h_out = int(h_in * scale)
        w_out = int(w_in * scale)

    y_out, x_out = np.meshgrid(np.arange(h_out), np.arange(w_out), indexing='ij')
    cx_in, cy_in = w_in / 2, h_in / 2
    cx_out, cy_out = w_out / 2, h_out / 2
    x_rel = (x_out - cx_out).astype(np.float32)
    y_rel = (y_out - cy_out).astype(np.float32)

    if angle != 0.0:
        rad = radians(-angle)
        x_rot = cos(rad) * x_rel - sin(rad) * y_rel
        y_rot = sin(rad) * x_rel + cos(rad) * y_rel
        x_rel, y_rel = x_rot + cx_in, y_rot + cy_in
    else:
        x_rel = x_rel / scale + cx_in
        y_rel = y_rel / scale + cy_in

    return interp_func(img, x_rel, y_rel)

def main():
    parser = argparse.ArgumentParser(description="Geometric Transformation Program")
    parser.add_argument('-a', '--angle', type=float, default=0.0, help="Ângulo de rotação em graus")
    parser.add_argument('-e', '--scale', type=float, default=1.0, help="Fator de escala")
    parser.add_argument('-d', '--dimension', type=int, nargs=2, help="Tamanho de saída: largura altura")
    parser.add_argument('-m', '--method', type=str, required=True, choices=['nearest', 'bilinear', 'bicubic', 'lagrange'], help="Método de interpolação")
    parser.add_argument('-i', '--input', type=str, required=True, help="Imagem de entrada (PNG)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Imagem de saída (PNG)")

    args = parser.parse_args()

    img = io.imread(args.input)
    out_size = tuple(args.dimension) if args.dimension else None

    result = transform_image(
        img,
        scale=args.scale,
        angle=args.angle,
        output_size=out_size,
        method=args.method
    )

    io.imsave(args.output, result)

if __name__ == "__main__":
    main()
