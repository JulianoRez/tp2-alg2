#!/usr/bin/env bash
# Todos os experimentos (retomável):  setsid nohup ./run_all.sh > results/run_all.log 2>&1 &
# Os B&B com 1-árvore vêm do lote results/v1 e não são refeitos aqui.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results
echo "início: $(date)"

# medições sensíveis a tempo: uma de cada vez, com a máquina parada
python run_experiments.py --no-bnb --workers 1 > results/heuristics.log 2>&1
echo "heurísticas: $(date)"
[ -f results/scaling.csv ] || python profile_scaling.py > results/scaling.log 2>&1
echo "escala (GC ligado): $(date)"
[ -f results/scaling_nogc.csv ] || python profile_scaling.py --no-gc --out results/scaling_nogc.csv > results/scaling_nogc.log 2>&1
echo "escala (GC desligado): $(date)"
[ -f results/audit.csv ] || python audit_naive.py > results/audit.log 2>&1
echo "auditoria: $(date)"

# B&B Held-Karp na TSPLIB (4 processos), com e sem GC
python run_experiments.py --no-heuristics --configs hk:bound hk:index --workers 4 \
    --trace-memory > results/bnb_hk.log 2>&1
echo "B&B Held-Karp: $(date)"
python run_experiments.py --no-heuristics --configs hk:bound --no-gc --workers 4 > results/bnb_nogc.log 2>&1
echo "B&B sem GC: $(date)"

# o que só depende de nós e custos, em paralelo
escala_bnb() {  # escala_bnb <limite> <tamanhos> <sementes> <sufixo>
    local out=results/scaling_bnb_$1_$4.csv
    [ -f "$out" ] || python profile_scaling.py --sizes --bnb-sizes $2 --seeds $3 \
        --bnb-bound "$1" --bnb-time 300 --out "$out" > "${out%.csv}.log" 2>&1
}
escala_bnb hk "10 15 20 25 30 35 40 45 50" "0 1 2 3 4" a &
escala_bnb hk "10 15 20 25 30 35 40 45 50" "5 6 7 8 9" b &
escala_bnb onetree "8 10 12 14 16 18 20 22 24 26" "0 1 2 3 4" a &
escala_bnb onetree "8 10 12 14 16 18 20 22 24 26" "5 6 7 8 9" b &
python run_synthetic.py --workers 4 >> results/synthetic.log 2>&1 &
wait
python run_synthetic.py --refeatures --workers 8 >> results/synthetic.log 2>&1
echo "escala do B&B + sintéticas: $(date)"

python run_experiments.py --csv-only > /dev/null 2>&1
python export_sheets.py > results/export.log 2>&1
echo "fim: $(date)"
