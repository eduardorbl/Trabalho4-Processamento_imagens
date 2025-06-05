#!/bin/bash

INPUT="imgs/objetos1.png"
OUTDIR="imgs2_output"
mkdir -p "$OUTDIR"

echo "Executando testes..."

# Teste 1: Nearest, escala 0.5
python session2/question5.py -e 0.5 -m nearest -i $INPUT -o $OUTDIR/saida_nearest_05.png

# Teste 2: Bilinear, escala 1.5
python session2/question5.py -e 1.5 -m bilinear -i $INPUT -o $OUTDIR/saida_bilinear_15.png

# Teste 3: Bicubic, escala 2.0, rotação 45
python session2/question5.py -e 2.0 -a 45 -m bicubic -i $INPUT -o $OUTDIR/saida_bicubic_2_rot45.png

# Teste 4: Lagrange, escala 10
python session2/question5.py -e 10.0 -m lagrange -i $INPUT -o $OUTDIR/saida_lagrange_10.png

# Teste 5: Nearest, rotação 90, escala 1.0
python session2/question5.py -e 1.0 -a 90 -m nearest -i $INPUT -o $OUTDIR/saida_nearest_rot90.png

# Teste 6: Bilinear com dimensão fixa 200x200
python session2/question5.py -d 200 200 -m bilinear -i $INPUT -o $OUTDIR/saida_bilinear_dim200x200.png

# Teste 7: Bicubic, sem escala, ângulo 180
python session2/question5.py -e 1.0 -a 180 -m bicubic -i $INPUT -o $OUTDIR/saida_bicubic_rot180.png

# Teste 8: Lagrange com dimensão fixa 400x400
python session2/question5.py -d 400 400 -m lagrange -i $INPUT -o $OUTDIR/saida_lagrange_dim400x400.png

# Teste 9: Nearest, imagem invertida (rotação 270)
python session2/question5.py -a 270 -e 1.0 -m nearest -i $INPUT -o $OUTDIR/saida_nearest_rot270.png

# Teste 10: Bicubic, escala muito pequena
python session2/question5.py -e 0.1 -m bicubic -i $INPUT -o $OUTDIR/saida_bicubic_01.png

echo "Todos os testes finalizados."
