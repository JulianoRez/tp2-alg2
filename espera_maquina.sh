#!/usr/bin/env bash
# Espera a máquina ficar livre (sem medições de outro projeto e carga < 1,5)
# e então roda ./run_all.sh.
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
