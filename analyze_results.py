#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ и визуализация результатов BehaviorSpace для модели MANET.

Принимает CSV-выгрузку BehaviorSpace в формате Table (Tools -> BehaviorSpace ->
Run -> Table output), усредняет каждую метрику по повторам для каждой пары
(протокол, значение варьируемого параметра), считает 95% доверительный интервал
и строит графики сравнения четырёх протоколов, а также сводные таблицы (CSV).

Использование:
    python3 analyze_results.py <behaviorspace_table.csv> [папка_вывода] [--watermark "текст"]

Варьируемый параметр серии (max-speed / num-nodes / num-flows / ...) определяется
автоматически как переменная (кроме routing-protocol) с более чем одним значением.
"""

import sys
import os
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

# --- подписи метрик (ось Y) ---
METRIC_LABELS = {
    "PDR": "Доля доставки PDR, %",
    "packet-loss-rate": "Потери пакетов, %",
    "avg-delay-ms": "Средняя задержка, мс",
    "max-delay-ms": "Максимальная задержка, мс",
    "throughput-bps": "Пропускная способность, бит/с",
    "network-utilization": "Утилизация сети",
    "NRO": "Норм. накладные расходы NRO",
    "control-packet-count": "Число служебных пакетов",
    "avg-route-discovery-ms": "Время обнаружения маршрута, мс",
    "avg-hop-count": "Среднее число переходов",
    "avg-residual-energy": "Средняя остаточная энергия, Дж",
    "alive-nodes": "Число живых узлов",
}

# --- подписи варьируемых параметров (ось X) ---
XLABELS = {
    "max-speed": "Максимальная скорость узлов, м/с",
    "num-nodes": "Число узлов",
    "num-flows": "Число CBR-потоков",
    "tx-radius": "Радиус передачи, м",
    "traffic-rate": "Интенсивность, пак/с",
    "pause-max": "Время паузы RWP, с",
    "link-failure-prob": "Вероятность отказа канала",
}

PROTO_ORDER = ["DQN-Routing", "Q-Routing", "AODV", "DSDV"]
PROTO_STYLE = {
    "DQN-Routing": ("#1f77b4", "o"),
    "Q-Routing":   ("#2ca02c", "s"),
    "AODV":        ("#ff7f0e", "^"),
    "DSDV":        ("#d62728", "D"),
}


def read_table(path):
    """Прочитать Table-вывод BehaviorSpace. Возвращает (header, список_строк-словарей, имя_эксперимента)."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    exp_name = ""
    # имя эксперимента обычно в 3-й строке метаданных
    if len(rows) >= 3 and rows[2]:
        exp_name = rows[2][0].strip()
    hi = None
    for i, r in enumerate(rows):
        if r and r[0].strip().lower() == "[run number]":
            hi = i
            break
    if hi is None:
        sys.exit("Не найдена строка заголовка '[run number]'. "
                 "Убедитесь, что это Table-вывод BehaviorSpace, а не Spreadsheet.")
    header = [c.strip() for c in rows[hi]]
    data = []
    for r in rows[hi + 1:]:
        if not r or len(r) < len(header):
            continue
        data.append(dict(zip(header, [c.strip() for c in r])))
    return header, data, exp_name


def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def ci95(vals):
    """Среднее и полуширина 95% доверительного интервала (t-распределение)."""
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(np.mean(vals))
    if n == 1:
        return m, 0.0, 1
    sd = float(np.std(vals, ddof=1))
    sem = sd / math.sqrt(n)
    t = float(sps.t.ppf(0.975, n - 1))
    return m, t * sem, n


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    watermark = None
    for a in sys.argv[1:]:
        if a.startswith("--watermark"):
            # формат --watermark=текст или следующий аргумент
            if "=" in a:
                watermark = a.split("=", 1)[1]
    if "--watermark" in sys.argv:
        idx = sys.argv.index("--watermark")
        if idx + 1 < len(sys.argv):
            watermark = sys.argv[idx + 1]

    if not args:
        print(__doc__)
        return
    path = args[0]
    outdir = args[1] if len(args) > 1 else "analysis_out"
    os.makedirs(outdir, exist_ok=True)

    header, data, exp_name = read_table(path)
    if "[step]" not in header:
        sys.exit("В заголовке нет столбца [step].")
    step_idx = header.index("[step]")
    var_cols = header[1:step_idx]            # переменные между [run number] и [step]
    metric_cols = header[step_idx + 1:]      # метрики после [step]

    # оставить строку финального шага для каждого прогона
    by_run = {}
    for d in data:
        rn = d.get("[run number]")
        st = to_float(d.get("[step]"))
        if (rn not in by_run) or (st > to_float(by_run[rn].get("[step]"))):
            by_run[rn] = d
    rundata = list(by_run.values())
    if not rundata:
        sys.exit("Нет строк с данными.")

    # варьируемый параметр (кроме routing-protocol) с >1 значением
    varied = None
    for c in var_cols:
        if c == "routing-protocol":
            continue
        if len({d.get(c) for d in rundata}) > 1:
            varied = c
            break

    protocols_present = [p for p in PROTO_ORDER
                         if p in {d.get("routing-protocol") for d in rundata}]
    if not protocols_present:
        protocols_present = sorted({d.get("routing-protocol") for d in rundata})

    title_suffix = (" — " + exp_name) if exp_name else ""

    # ----- агрегирование -----
    # groups[(proto, xval)][metric] = список значений
    groups = defaultdict(lambda: defaultdict(list))
    if varied is not None:
        xvals = sorted({to_float(d.get(varied)) for d in rundata})
        for d in rundata:
            key = (d.get("routing-protocol"), to_float(d.get(varied)))
            for mtr in metric_cols:
                groups[key][mtr].append(to_float(d.get(mtr)))
    else:
        # варьируется только протокол — построим столбчатую диаграмму
        xvals = [0]
        for d in rundata:
            key = (d.get("routing-protocol"), 0)
            for mtr in metric_cols:
                groups[key][mtr].append(to_float(d.get(mtr)))

    # ----- сводная таблица (длинный формат) -----
    long_path = os.path.join(outdir, "summary_long.csv")
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", varied or "(только протокол)", "metric", "mean", "ci95", "n"])
        for proto in protocols_present:
            for xv in xvals:
                for mtr in metric_cols:
                    m, e, n = ci95(groups[(proto, xv)][mtr])
                    if n > 0:
                        w.writerow([proto, xv, mtr, f"{m:.4f}", f"{e:.4f}", n])

    # ----- сводные таблицы по каждой метрике (mean ± ci) -----
    for mtr in metric_cols:
        tpath = os.path.join(outdir, "table_%s.csv" % safe(mtr))
        with open(tpath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([varied or "param"] + protocols_present)
            for xv in xvals:
                row = [fmt_x(xv)]
                for proto in protocols_present:
                    m, e, n = ci95(groups[(proto, xv)][mtr])
                    row.append("" if n == 0 else f"{m:.3f} ± {e:.3f}")
                w.writerow(row)

    # ----- графики -----
    n_plots = 0
    for mtr in metric_cols:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plotted = False
        if varied is not None:
            for proto in protocols_present:
                xs, ys, es = [], [], []
                for xv in xvals:
                    m, e, n = ci95(groups[(proto, xv)][mtr])
                    if n > 0:
                        xs.append(xv); ys.append(m); es.append(e)
                if xs:
                    color, marker = PROTO_STYLE.get(proto, ("#444444", "o"))
                    ax.errorbar(xs, ys, yerr=es, label=proto, color=color, marker=marker,
                                markersize=6, capsize=3, linewidth=2)
                    plotted = True
            ax.set_xlabel(XLABELS.get(varied, varied))
        else:
            means, errs, labels, colors = [], [], [], []
            for proto in protocols_present:
                m, e, n = ci95(groups[(proto, 0)][mtr])
                if n > 0:
                    means.append(m); errs.append(e); labels.append(proto)
                    colors.append(PROTO_STYLE.get(proto, ("#444444", "o"))[0])
            if means:
                ax.bar(labels, means, yerr=errs, capsize=4, color=colors)
                plotted = True
            ax.set_xlabel("Протокол")

        ax.set_ylabel(METRIC_LABELS.get(mtr, mtr))
        ax.set_title(METRIC_LABELS.get(mtr, mtr) + title_suffix)
        ax.grid(True, alpha=0.3)
        if varied is not None and plotted:
            ax.legend(title="Протокол")
        if watermark:
            ax.text(0.5, 0.5, watermark, transform=ax.transAxes, fontsize=20,
                    color="red", alpha=0.18, ha="center", va="center", rotation=25,
                    fontweight="bold")
        fig.tight_layout()
        if plotted:
            fig.savefig(os.path.join(outdir, "plot_%s.png" % safe(mtr)), dpi=130)
            n_plots += 1
        plt.close(fig)

    print("Эксперимент:", exp_name or "(имя не найдено)")
    print("Варьируемый параметр:", varied or "(только протокол)")
    print("Протоколы:", ", ".join(protocols_present))
    print("Прогонов (строк):", len(rundata))
    print("Метрик:", len(metric_cols))
    print("Сохранено графиков:", n_plots)
    print("Папка результатов:", outdir)


def safe(s):
    return "".join(c if c.isalnum() else "_" for c in s)


def fmt_x(x):
    if float(x).is_integer():
        return str(int(x))
    return str(x)


if __name__ == "__main__":
    main()
