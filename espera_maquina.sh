#!/usr/bin/env bash
# Espera a máquina ficar livre e então roda ./run_all.sh.
# "Livre" = nenhuma medição do projeto ciclos-temporais rodando (medir*.sh,
# kdfs, ciclos_gpu, cycle da IBM) e carga média de 1 min abaixo de 1,5,
# por 2 checagens seguidas. Medições de tempo das duas não podem se misturar.
cd "$(dirname "$0")"
ocupada() {
    pgrep -f 'medir(_completo)?\.sh|build/kdfs|build/ciclos_gpu|build/cycle' > /dev/null && return 0
    awk '{exit !($1 >= 1.5)}' /proc/loadavg
}
echo "esperando a máquina: $(date)"
livres=0
while [ $livres -lt 2 ]; do
    if ocupada; then livres=0; else livres=$((livres + 1)); fi
    sleep 60
done
echo "máquina livre: $(date) carga: $(cat /proc/loadavg)"
exec ./run_all.sh
