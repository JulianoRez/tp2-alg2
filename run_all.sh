#!/usr/bin/env bash
# Roda todos os experimentos do trabalho (versão 2, depois das correções).
#   setsid nohup ./run_all.sh > results/run_all.log 2>&1 &
# Retomável: rodar de novo pula o que já terminou (runs.jsonl, synthetic.jsonl)
# e os CSVs que já existem.
#
# Os B&B com 1-árvore (150 execuções, quase metade estoura 30 min) NÃO são
# refeitos: vêm do lote v1 (results/v1, 8 processos em paralelo). O código da
# DFS não mudou desde então, e a contagem de nós de quem terminou é
# determinística.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results
echo "início: $(date)"

# --- Fase A: medições sensíveis a tempo, UMA de cada vez, máquina parada ----
python run_experiments.py --no-bnb --workers 1 > results/heuristics.log 2>&1
echo "heurísticas: $(date)"
[ -f results/scaling.csv ] || python profile_scaling.py > results/scaling.log 2>&1
echo "escala (GC ligado): $(date)"
[ -f results/scaling_nogc.csv ] || python profile_scaling.py --no-gc --out results/scaling_nogc.csv > results/scaling_nogc.log 2>&1
echo "escala (GC desligado): $(date)"
[ -f results/audit.csv ] || python audit_naive.py > results/audit.log 2>&1
echo "auditoria: $(date)"

# --- Fase B1: B&B Held-Karp na TSPLIB, 4 processos, sem mais nada rodando ---
python run_experiments.py --no-heuristics --configs hk:bound hk:index --workers 4 \
    --trace-memory > results/bnb_hk.log 2>&1
echo "B&B Held-Karp: $(date)"
# mesmo B&B com o GC desligado, nas mesmas condições (o GC distorce o B&B?)
python run_experiments.py --no-heuristics --configs hk:bound --no-gc --workers 4 > results/bnb_nogc.log 2>&1
echo "B&B sem GC: $(date)"

# --- Fase B2: o que só depende de contagem de nós / custo, em paralelo -----
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
