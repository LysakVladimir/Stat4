# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class TrendResult:
    name: str
    equation: str
    parameters: Dict[str, float]
    fitted: List[float]
    r2: float
    adj_r2: float
    rmse: float
    mae: float
    mape: float


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    result: List[str] = []
    for si in root.findall("a:si", NS):
        texts = []
        for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
            texts.append(node.text or "")
        result.append("".join(texts))
    return result


def _cell_to_indexes(ref: str) -> Tuple[int, int]:
    match = re.match(r"([A-Z]+)(\d+)", ref)
    if not match:
        raise ValueError(f"Некорректная ссылка на ячейку: {ref}")
    letters, row_number = match.groups()
    column = 0
    for ch in letters:
        column = column * 26 + ord(ch) - 64
    return int(row_number) - 1, column - 1


def _read_xlsx_sheet(path: str, sheet_xml: str) -> List[List[Any]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        root = ET.fromstring(zf.read(sheet_xml))
    rows: List[List[Any]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: Dict[int, Any] = {}
        max_col = 0
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "A1")
            _, col = _cell_to_indexes(ref)
            max_col = max(max_col, col)
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", NS)
            if value_node is None or value_node.text is None:
                value: Any = None
            else:
                raw = value_node.text
                if cell_type == "s":
                    value = shared_strings[int(raw)]
                elif cell_type == "b":
                    value = bool(int(raw))
                else:
                    try:
                        value = float(raw) if any(c in raw for c in ".Ee") else int(raw)
                    except ValueError:
                        value = raw
            values[col] = value
        rows.append([values.get(i) for i in range(max_col + 1)])
    return rows


def load_un_tourism_data(xlsx_path: str) -> List[Dict[str, Any]]:
    """Читает лист Data из книги UN Tourism и возвращает список словарей."""
    rows = _read_xlsx_sheet(xlsx_path, "xl/worksheets/sheet2.xml")
    header = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [None] * (len(header) - len(row))
        records.append(dict(zip(header, padded)))
    return records


def select_series(
    records: List[Dict[str, Any]],
    country: str = "Portugal",
    indicator_code: str = "INBD_TRIP_TOTL_TOTL_TOUR",
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str | None]]:
    selected = [
        r
        for r in records
        if r.get("reporter_area_label") == country
        and r.get("indicator_code") == indicator_code
        and isinstance(r.get("year"), int)
        and r.get("value") is not None
    ]
    selected.sort(key=lambda r: int(r["year"]))
    if len(selected) < 15:
        raise ValueError("Выбранный ряд содержит менее 15 уровней")
    metadata = {
        "indicator_code": selected[0]["indicator_code"],
        "indicator_label": selected[0]["indicator_label"],
        "country": selected[0]["reporter_area_label"],
        "partner_area": selected[0]["partner_area_label"],
        "unit": selected[0]["unit"],
        "source_dataset": "All Countries: Inbound Tourism: Arrivals 1995 - 2024 (12.2025)",
    }
    years = np.array([int(r["year"]) for r in selected], dtype=int)
    values = np.array([float(r["value"]) for r in selected], dtype=float)
    flags = [r.get("flag_label") for r in selected]
    return metadata, years, values, flags


def descriptive_statistics(years: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    return {
        "n": float(n),
        "first_year": float(years[0]),
        "last_year": float(years[-1]),
        "initial": float(values[0]),
        "final": float(values[-1]),
        "absolute_change": float(values[-1] - values[0]),
        "growth_factor": float(values[-1] / values[0]),
        "cagr_percent": float(((values[-1] / values[0]) ** (1 / (n - 1)) - 1) * 100),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "cv_percent": float(np.std(values, ddof=1) / np.mean(values) * 100),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def autocorrelation(values: np.ndarray, max_lag: int = 10) -> List[float]:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    denominator = np.sum((values - mean) ** 2)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        numerator = np.sum((values[lag:] - mean) * (values[:-lag] - mean))
        acf.append(float(numerator / denominator))
    return acf


def stationarity_by_halves(values: np.ndarray) -> Dict[str, float]:
    half = len(values) // 2
    first = values[:half]
    second = values[half:]
    t_test = scipy_stats.ttest_ind(second, first, equal_var=False)
    var_first = np.var(first, ddof=1)
    var_second = np.var(second, ddof=1)
    f_ratio = max(var_first, var_second) / min(var_first, var_second)
    df1 = len(first) - 1
    df2 = len(second) - 1
    p_f = 2 * min(scipy_stats.f.cdf(f_ratio, df1, df2), 1 - scipy_stats.f.cdf(f_ratio, df1, df2))
    return {
        "mean_first": float(np.mean(first)),
        "mean_second": float(np.mean(second)),
        "std_first": float(np.std(first, ddof=1)),
        "std_second": float(np.std(second, ddof=1)),
        "t_stat": float(t_test.statistic),
        "t_pvalue": float(t_test.pvalue),
        "f_ratio": float(f_ratio),
        "f_pvalue": float(p_f),
    }


def irwin_anomalies(years: np.ndarray, values: np.ndarray, threshold: float = 1.5) -> List[Dict[str, float]]:
    s = np.std(values, ddof=1)
    anomalies = []
    for i in range(1, len(values)):
        lam = abs(values[i] - values[i - 1]) / s
        if lam > threshold:
            anomalies.append(
                {
                    "year": int(years[i]),
                    "value": float(values[i]),
                    "lambda": float(lam),
                    "change": float(values[i] - values[i - 1]),
                }
            )
    return anomalies


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = window // 2
    for i in range(half, len(values) - half):
        result[i] = np.mean(values[i - half : i + half + 1])
    return result


def weighted_moving_average(values: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    weights = np.asarray(list(weights), dtype=float)
    weights = weights / weights.sum()
    window = len(weights)
    result = np.full(len(values), np.nan)
    half = window // 2
    for i in range(half, len(values) - half):
        result[i] = np.sum(values[i - half : i + half + 1] * weights)
    return result


def additive_decomposition(values: np.ndarray, period: int = 5) -> Dict[str, Any]:
    trend = moving_average(values, period)
    detrended = values - trend
    pattern = []
    for position in range(period):
        idx = [i for i in range(len(values)) if not np.isnan(trend[i]) and i % period == position]
        pattern.append(float(np.nanmean(detrended[idx])) if idx else 0.0)
    pattern = np.array(pattern)
    pattern = pattern - np.nanmean(pattern)
    periodic = np.array([pattern[i % period] for i in range(len(values))])
    residual = values - trend - periodic
    return {"trend": trend, "periodic": periodic, "pattern": pattern, "residual": residual}


def _quality_metrics(values: np.ndarray, fitted: np.ndarray, parameter_count: int) -> Dict[str, float]:
    residual = values - fitted
    sse = np.sum(residual**2)
    sst = np.sum((values - values.mean()) ** 2)
    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (len(values) - 1) / (len(values) - parameter_count)
    return {
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual / values)) * 100),
    }


def fit_trends(values: np.ndarray) -> Dict[str, TrendResult]:
    t = np.arange(1, len(values) + 1, dtype=float)
    results: Dict[str, TrendResult] = {}

    # Linear model y = a + b*t
    b, a = np.polyfit(t, values, 1)
    fitted = a + b * t
    metrics = _quality_metrics(values, fitted, 2)
    results["linear"] = TrendResult("Линейная", f"y = {a:.3f} + {b:.3f}t", {"a": a, "b": b}, fitted.tolist(), **metrics)

    # Quadratic model y = a + b*t + c*t^2
    c, b, a = np.polyfit(t, values, 2)
    fitted = a + b * t + c * t**2
    metrics = _quality_metrics(values, fitted, 3)
    results["quadratic"] = TrendResult(
        "Параболическая", f"y = {a:.3f} + {b:.3f}t + {c:.3f}t²", {"a": a, "b": b, "c": c}, fitted.tolist(), **metrics
    )

    # Cubic polynomial model y = a + b*t + c*t^2 + d*t^3
    d, c, b, a = np.polyfit(t, values, 3)
    fitted = a + b * t + c * t**2 + d * t**3
    metrics = _quality_metrics(values, fitted, 4)
    results["cubic"] = TrendResult(
        "Полином 3-й степени",
        f"y = {a:.3f} + {b:.3f}t + {c:.3f}t² + {d:.3f}t³",
        {"a": a, "b": b, "c": c, "d": d},
        fitted.tolist(),
        **metrics,
    )

    # Exponential model y = a*b^t, estimated through logarithms
    log_y = np.log(values)
    log_b, log_a = np.polyfit(t, log_y, 1)
    a = math.exp(log_a)
    b = math.exp(log_b)
    fitted = a * (b**t)
    metrics = _quality_metrics(values, fitted, 2)
    results["exponential"] = TrendResult(
        "Экспоненциальная", f"y = {a:.3f} · {b:.6f}ᵗ", {"a": a, "b": b}, fitted.tolist(), **metrics
    )

    # Hyperbolic model y = a + b/t
    x = np.column_stack([np.ones_like(t), 1 / t])
    a, b = np.linalg.lstsq(x, values, rcond=None)[0]
    fitted = a + b / t
    metrics = _quality_metrics(values, fitted, 2)
    results["hyperbolic"] = TrendResult("Гиперболическая", f"y = {a:.3f} + {b:.3f}/t", {"a": a, "b": b}, fitted.tolist(), **metrics)

    # Logistic model y = k / (1 + b*exp(-a*t))
    def logistic(x: np.ndarray, k: float, b: float, a: float) -> np.ndarray:
        return k / (1 + b * np.exp(-a * x))

    try:
        p0 = [max(values) * 1.5, 4.0, 0.08]
        lower = [max(values) * 0.8, 0.0, 0.0]
        upper = [max(values) * 10, 1000.0, 5.0]
        k, b, a = curve_fit(logistic, t, values, p0=p0, bounds=(lower, upper), maxfev=100000)[0]
        fitted = logistic(t, k, b, a)
        metrics = _quality_metrics(values, fitted, 3)
        results["logistic"] = TrendResult(
            "Логистическая",
            f"y = {k:.3f} / (1 + {b:.3f}e^(-{a:.5f}t))",
            {"k": k, "b": b, "a": a},
            fitted.tolist(),
            **metrics,
        )
    except Exception:
        pass

    return results


def residual_diagnostics(values: np.ndarray, fitted: np.ndarray) -> Dict[str, float]:
    residual = values - fitted
    n = len(residual)
    mean = np.mean(residual)
    std = np.std(residual, ddof=1)
    dw = np.sum(np.diff(residual) ** 2) / np.sum(residual**2)
    acf1 = np.corrcoef(residual[1:], residual[:-1])[0, 1]

    median = np.median(residual)
    signs = [1 if r > median else -1 if r < median else 0 for r in residual]
    signs = [s for s in signs if s != 0]
    runs = 1 + sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1]) if signs else 0
    max_run = 0
    current = 0
    previous = None
    for sign in signs:
        current = current + 1 if sign == previous else 1
        previous = sign
        max_run = max(max_run, current)
    runs_threshold = 0.5 * (len(signs) + 1 - 1.96 * math.sqrt(len(signs) - 1))
    max_run_threshold = 3.3 * (math.log10(len(signs)) + 1)

    turning_points = 0
    for i in range(1, n - 1):
        if (residual[i - 1] < residual[i] > residual[i + 1]) or (residual[i - 1] > residual[i] < residual[i + 1]):
            turning_points += 1
    turning_mean = 2 / 3 * (n - 2)
    turning_sigma = math.sqrt((16 * n - 29) / 90)
    turning_lower = turning_mean - 1.96 * turning_sigma

    t_zero = mean / (std / math.sqrt(n)) if std > 0 else 0.0
    p_zero = 2 * (1 - scipy_stats.t.cdf(abs(t_zero), df=n - 1))

    m2 = np.mean(residual**2)
    asymmetry = float((np.mean(residual**3)) / (m2 ** 1.5)) if m2 > 0 else 0.0
    excess = float((np.mean(residual**4)) / (m2**2) - 3) if m2 > 0 else 0.0
    sigma_asymmetry = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    sigma_excess = math.sqrt(24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5)))

    return {
        "mean": float(mean),
        "std": float(std),
        "durbin_watson": float(dw),
        "residual_acf_lag1": float(acf1),
        "runs": float(runs),
        "max_run": float(max_run),
        "runs_threshold": float(runs_threshold),
        "max_run_threshold": float(max_run_threshold),
        "turning_points": float(turning_points),
        "turning_lower": float(turning_lower),
        "t_zero": float(t_zero),
        "p_zero": float(p_zero),
        "asymmetry": float(asymmetry),
        "excess": float(excess),
        "sigma_asymmetry": float(sigma_asymmetry),
        "sigma_excess": float(sigma_excess),
    }


def make_figures(years: np.ndarray, values: np.ndarray, acf: List[float], decomposition: Dict[str, Any], trends: Dict[str, TrendResult], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # Обновленная контрастная палитра отчета: графитовый, сине-зеленый, терракотовый, сливовый и светло-серый.
    palette = {
        "primary": "#263238",
        "secondary": "#007C89",
        "accent": "#C65D3B",
        "orange": "#D99A24",
        "purple": "#5E548E",
        "gray": "#6B7280",
        "light_gray": "#C9CED6",
        "positive": "#007C89",
        "negative": "#9D3C3C",
        "zero": "#30343F",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": palette["gray"],
        "grid.color": "#D9DDE3",
    })
    paths: Dict[str, str] = {}

    def save(name: str) -> str:
        path = os.path.join(out_dir, f"{name}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close()
        paths[name] = path
        return path

    plt.figure(figsize=(8.2, 4.1))
    plt.plot(years, values, marker="o", linewidth=2, color=palette["primary"], markerfacecolor=palette["secondary"], markeredgecolor=palette["primary"])
    plt.title("Ряд международных прибытий ночующих туристов: Португалия")
    plt.xlabel("Год")
    plt.ylabel("Значение, тыс. поездок")
    plt.grid(True, alpha=0.3)
    save("fig1_dynamic")

    plt.figure(figsize=(8.2, 4.1))
    plt.bar(years, values, color=palette["secondary"], edgecolor=palette["primary"], linewidth=0.6)
    plt.title("Годовые уровни и резкие перепады ряда")
    plt.xlabel("Год")
    plt.ylabel("Значение, тыс. поездок")
    plt.xticks(years[::2], rotation=45)
    plt.grid(axis="y", alpha=0.3)
    save("fig2_bar")

    lags = np.arange(len(acf))
    signif = 2 / math.sqrt(len(values))
    plt.figure(figsize=(8.2, 4.1))
    plt.axhline(0, linewidth=1, color=palette["zero"])
    plt.axhline(signif, linestyle="--", linewidth=1, color=palette["accent"])
    plt.axhline(-signif, linestyle="--", linewidth=1, color=palette["accent"])
    plt.vlines(lags, 0, acf, linewidth=2, color=palette["primary"])
    plt.scatter(lags, acf, zorder=3, color=palette["secondary"], edgecolor=palette["primary"])
    plt.title("Автокорреляционный профиль ряда")
    plt.xlabel("Лаг")
    plt.ylabel("Автокорреляция")
    plt.xticks(lags)
    plt.grid(True, axis="y", alpha=0.3)
    save("fig3_correlogram")

    plt.figure(figsize=(8.2, 4.1))
    plt.plot(years, values, marker="o", alpha=0.45, label="Исходный ряд", color=palette["gray"], markerfacecolor=palette["light_gray"], markeredgecolor=palette["gray"])
    plt.plot(years, decomposition["trend"], marker="o", linewidth=2.2, label="Трендовая компонента", color=palette["secondary"], markerfacecolor=palette["secondary"], markeredgecolor=palette["primary"])
    plt.title("Сглаженная траектория основного движения")
    plt.xlabel("Год")
    plt.ylabel("Значение, тыс. поездок")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("fig4_trend_component")

    plt.figure(figsize=(8.2, 3.7))
    plt.plot(years, decomposition["periodic"], marker="o", linewidth=2, color=palette["purple"], markerfacecolor=palette["orange"], markeredgecolor=palette["purple"])
    plt.axhline(0, linewidth=1, color=palette["zero"])
    plt.title("Условная повторяющаяся компонента")
    plt.xlabel("Год")
    plt.ylabel("Отклонение, тыс. поездок")
    plt.grid(True, alpha=0.3)
    save("fig5_periodic_component")

    plt.figure(figsize=(8.2, 3.7))
    residual_colors = [palette["positive"] if x >= 0 else palette["negative"] for x in decomposition["residual"]]
    plt.bar(years, decomposition["residual"], color=residual_colors, edgecolor=palette["zero"], linewidth=0.4)
    plt.axhline(0, linewidth=1, color=palette["zero"])
    plt.title("Отклонения после декомпозиции")
    plt.xlabel("Год")
    plt.ylabel("Остаток, тыс. поездок")
    plt.xticks(years[::2], rotation=45)
    plt.grid(axis="y", alpha=0.3)
    save("fig6_decomposition_residual")

    ma3 = moving_average(values, 3)
    ma5 = moving_average(values, 5)
    wma5 = weighted_moving_average(values, [1, 2, 3, 2, 1])
    plt.figure(figsize=(8.2, 4.1))
    plt.plot(years, values, marker="o", alpha=0.35, label="Исходный ряд", color=palette["gray"], markerfacecolor=palette["light_gray"], markeredgecolor=palette["gray"])
    plt.plot(years, ma3, linewidth=2, label="Скользящая средняя, m=3", color=palette["primary"])
    plt.plot(years, ma5, linewidth=2, label="Скользящая средняя, m=5", color=palette["orange"])
    plt.plot(years, wma5, linewidth=2, label="Взвешенная средняя, m=5", color=palette["secondary"])
    plt.title("Сопоставление сглаживающих фильтров")
    plt.xlabel("Год")
    plt.ylabel("Значение, тыс. поездок")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("fig7_smoothing")

    t = np.arange(1, len(values) + 1, dtype=float)
    plt.figure(figsize=(8.2, 4.1))
    plt.plot(years, values, marker="o", linewidth=2, label="Фактические уровни", color=palette["primary"], markerfacecolor=palette["secondary"], markeredgecolor=palette["primary"])
    trend_colors = {"linear": palette["gray"], "quadratic": palette["orange"], "cubic": palette["accent"], "exponential": palette["purple"]}
    for key in ["linear", "quadratic", "cubic", "exponential"]:
        tr = trends[key]
        plt.plot(years, tr.fitted, linewidth=1.8, label=tr.name, color=trend_colors[key])
    plt.title("Ряд и аналитические линии тренда")
    plt.xlabel("Год")
    plt.ylabel("Значение, тыс. поездок")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save("fig8_trend_models")

    best = trends["cubic"]
    residual = values - np.array(best.fitted)
    plt.figure(figsize=(8.2, 3.7))
    residual_colors = [palette["positive"] if x >= 0 else palette["negative"] for x in residual]
    plt.bar(years, residual, color=residual_colors, edgecolor=palette["zero"], linewidth=0.4)
    plt.axhline(0, linewidth=1, color=palette["zero"])
    plt.title("Ошибки выбранной кубической модели")
    plt.xlabel("Год")
    plt.ylabel("Остаток, тыс. поездок")
    plt.xticks(years[::2], rotation=45)
    plt.grid(axis="y", alpha=0.3)
    save("fig9_cubic_residuals")

    return paths


def run_all(xlsx_path: str, out_dir: str) -> Dict[str, Any]:
    records = load_un_tourism_data(xlsx_path)
    metadata, years, values, flags = select_series(records)
    desc = descriptive_statistics(years, values)
    acf = autocorrelation(values, 10)
    stationarity = stationarity_by_halves(values)
    anomalies = irwin_anomalies(years, values)
    decomposition = additive_decomposition(values, period=5)
    trends = fit_trends(values)
    diagnostics = residual_diagnostics(values, np.array(trends["cubic"].fitted))
    figures = make_figures(years, values, acf, decomposition, trends, os.path.join(out_dir, "figures"))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selected_series.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["year", "value", "unit", "flag_label"])
        for year, value, flag in zip(years, values, flags):
            writer.writerow([year, value, metadata["unit"], flag or ""])

    result = {
        "metadata": metadata,
        "years": years.tolist(),
        "values": values.tolist(),
        "flags": flags,
        "descriptive": desc,
        "autocorrelation": acf,
        "stationarity": stationarity,
        "anomalies_irwin": anomalies,
        "decomposition_pattern": decomposition["pattern"].tolist(),
        "decomposition_periodic_amplitude": float(np.nanmax(decomposition["periodic"]) - np.nanmin(decomposition["periodic"])),
        "trends": {key: asdict(value) for key, value in trends.items()},
        "residual_diagnostics": diagnostics,
        "figures": figures,
    }
    with open(os.path.join(out_dir, "practice4_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    input_path = "mnt\data\Tourism_inbound_arrivals_12_2025.xlsx"
    output_dir = os.path.join("practice4_calcs")
    data = run_all(input_path, output_dir)
    print(json.dumps({
        "country": data["metadata"]["country"],
        "indicator": data["metadata"]["indicator_code"],
        "years": [data["years"][0], data["years"][-1]],
        "levels": len(data["years"]),
        "best_model": "Полином 3-й степени",
        "output_dir": output_dir,
    }, ensure_ascii=False, indent=2))
