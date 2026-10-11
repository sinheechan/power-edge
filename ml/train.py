# 학습: 데이터 분할 -> 모델 6개 Optuna 튜닝·학습 -> 앙상블 전 조합 평가 -> CSV + .log + model.pkl
#
# 사용법 (ml 폴더에서):
#   python train.py                         # 6개 모델 전부, 모델당 Optuna 20회
#   python train.py --trials 3              # 빠르게 확인
#   python train.py --models ridge,lgbm,gru # 일부만
#   python train.py --include-db            # DB 실시간 데이터까지 포함 (serve.py 재학습용)
#
# 결과:
#   outputs/results.csv      기준 2 + 단일 6 + 앙상블 상위 5 (+선택된 앙상블) 누적
#   outputs/retrain_log.csv  기존 모델과 비교·교체 기록 누적
#   outputs/logs/train_<run_id>.log  실행 로그 (튜닝 시도, 단일 모델·앙상블 전 조합 결과와 파라미터)
#   models/model.pkl         선택된 앙상블 (기존 모델보다 좋을 때만 교체)

import argparse
import json
import os
from datetime import datetime
from itertools import combinations
from math import comb

import joblib
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import features
from features import TAB_COLS
from models import MODEL_NAMES, SEQ_MODELS, make_model, predict_bundle, suggest_params

OUT_DIR = os.path.join(features.ML_DIR, "outputs")
MODEL_PATH = os.path.join(features.ML_DIR, "models", "model.pkl")
TUNE_ROWS = 30000         # 튜닝은 학습 데이터 일부로 빠르게, 최종 학습은 전체로
TARGET_PCT = 95           # 평가용 목표전력 = 학습 구간 15분 평균의 상위 5% 선
SWITCH_GAIN = 0.02        # 4개 조합이 3개보다 2% 이상 좋을 때만 4개 선택

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG_PATH = None           # main() 에서 실행마다 정해짐


# 화면에 출력하면서 .log 파일에도 같은 내용을 기록
def log_write_print(message=""):
    print(message)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


# 결과 1줄 요약 (기준·단일 모델·앙상블 공통)
def result_text(row):
    return (f"valid {row['valid_mae']:7.1f} kW | test {row['test_mae']:7.1f} kW | "
            f"초반 {row['test_mae_early']:7.1f} kW | 놓친 피크 {row['missed_peaks']}/{row['peak_windows']} | "
            f"헛경보 {row['false_alarms']} | 조기 {row['early_min']:.1f}분")


# ---------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------
def load_data(include_db):
    log_write_print("CSV 로드 중...")
    steps, scale = features.load_csv_power()
    df, X_seq = features.build_samples(steps)
    log_write_print(f"  CSV 샘플 {len(df):,}개 (배율 x{scale:.1f})")

    # DB 데이터는 CSV 와 날짜가 떨어져 있어서 따로 샘플을 만든 뒤 합침
    if include_db:
        load_dotenv()
        db_steps = features.load_db_power(os.getenv("DB_URL"))
        if len(db_steps) > 0:
            db_df, db_seq = features.build_samples(db_steps)
            log_write_print(f"  DB 샘플 {len(db_df):,}개")
            if len(db_df) > 0:
                df = pd.concat([df, db_df], ignore_index=True)
                X_seq = np.concatenate([X_seq, db_seq])

    return df, X_seq, scale


# ---------------------------------------------------------------
# 평가
#   mae          : 전체 예측 오차 (kW)
#   mae_early    : 구간 초반(8분 이내) 오차 — ML 이 가장 도움 되는 구간
#   missed_peaks : 실제 목표 초과인데 구간 끝 3분 전까지 경보를 못 낸 구간 수
#   false_alarms : 실제 목표 90% 미만인데 경보를 낸 구간 수
#   early_min    : 초과 구간을 평균 몇 분 일찍 알아챘나 (클수록 좋음)
# ---------------------------------------------------------------
def evaluate(part, pred, target_kw, with_peaks):
    actual = part["actual_kw"].to_numpy()
    err = np.abs(pred - actual)
    early_mask = part["elapsed_min"].to_numpy() <= 8

    result = {
        "mae": float(err.mean()),
        "mae_early": float(err[early_mask].mean()),
    }
    if not with_peaks:
        return result

    tmp = pd.DataFrame({
        "window_start": part["window_start"].to_numpy(),
        "elapsed_min": part["elapsed_min"].to_numpy(),
        "actual": actual,
        "pred": pred,
    })

    missed = 0
    false_alarms = 0
    peak_windows = 0
    leads = []
    for ws, g in tmp.groupby("window_start"):
        window_actual = g["actual"].iloc[0]
        alarms = g[g["pred"] >= target_kw]

        if window_actual >= target_kw:
            peak_windows += 1
            in_time = alarms[alarms["elapsed_min"] <= 12]
            if len(in_time) == 0:
                missed += 1
            else:
                leads.append(15 - in_time["elapsed_min"].min())
        elif window_actual < target_kw * 0.9 and len(alarms) > 0:
            false_alarms += 1

    result["peak_windows"] = peak_windows
    result["missed_peaks"] = missed
    result["false_alarms"] = false_alarms
    result["early_min"] = float(np.mean(leads)) if len(leads) > 0 else 0.0
    return result


# ---------------------------------------------------------------
# 단일 모델: Optuna 튜닝 -> 전체 학습 데이터로 다시 학습
# ---------------------------------------------------------------
def tune_and_fit(name, data, trials):
    X_tr, S_tr, y_tr = data["train"]
    X_va, S_va, y_va = data["valid"]

    # 튜닝용 일부 샘플
    rng = np.random.default_rng(42)
    if len(y_tr) > TUNE_ROWS:
        pick = rng.choice(len(y_tr), TUNE_ROWS, replace=False)
    else:
        pick = np.arange(len(y_tr))

    def objective(trial):
        params = suggest_params(name, trial)
        model = make_model(name, params)
        model.fit(X_tr[pick], S_tr[pick], y_tr[pick])
        pred = model.predict(X_va, S_va)
        return float(np.mean(np.abs(pred - y_va)))

    # 딥러닝은 오래 걸려서 시도 횟수를 1/3 로
    n_trials = trials
    if name in SEQ_MODELS:
        n_trials = max(3, trials // 3)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    with tqdm(total=n_trials, desc=f"  {name} 튜닝", unit="회") as bar:
        def on_trial(study, trial):
            bar.update(1)
            bar.set_postfix(best=f"{study.best_value:.1f}kW")
        study.optimize(objective, n_trials=n_trials, callbacks=[on_trial])

    log_write_print(f"  [{name}] 튜닝 시도 {n_trials}회 (학습 샘플 {len(pick):,}개)")
    for t in study.trials:
        mark = " <- best" if t.number == study.best_trial.number else ""
        log_write_print(f"    trial {t.number:3d} | valid {t.value:7.1f} kW | {json.dumps(t.params)}{mark}")

    best = study.best_params
    model = make_model(name, best)
    model.fit(X_tr, S_tr, y_tr)
    return model, best


# ---------------------------------------------------------------
# 앙상블: 2~N개 전 조합 x (단순 평균, 가중 평균)
#   가중 평균 가중치 = valid 오차의 역수 (잘하는 모델에 더 큰 비중)
# ---------------------------------------------------------------
def make_ensembles(names, preds, valid_mae, parts, target_kw, run_id):
    rows = []
    total = sum(comb(len(names), n) for n in range(2, len(names) + 1)) * 2
    bar = tqdm(total=total, desc="  앙상블 평가", unit="개")
    for n in range(2, len(names) + 1):
        for combo in combinations(names, n):
            for method in ["mean", "weighted"]:
                if method == "mean":
                    weights = [1.0 / n] * n
                else:
                    inv = []
                    for m in combo:
                        inv.append(1.0 / valid_mae[m])
                    total = sum(inv)
                    weights = []
                    for v in inv:
                        weights.append(v / total)

                pred_va = np.zeros(len(parts["valid"]))
                pred_te = np.zeros(len(parts["test"]))
                for m, w in zip(combo, weights):
                    pred_va = pred_va + w * preds[m]["valid"]
                    pred_te = pred_te + w * preds[m]["test"]

                ev_va = evaluate(parts["valid"], pred_va, target_kw, with_peaks=False)
                ev_te = evaluate(parts["test"], pred_te, target_kw, with_peaks=True)

                weight_text = []
                for w in weights:
                    weight_text.append(f"{w:.2f}")

                rows.append({
                    "run_id": run_id,
                    "kind": "ensemble",
                    "method": method,
                    "n_models": n,
                    "members": "+".join(combo),
                    "weights": "/".join(weight_text),
                    "valid_mae": ev_va["mae"],
                    "test_mae": ev_te["mae"],
                    "test_mae_early": ev_te["mae_early"],
                    "missed_peaks": ev_te["missed_peaks"],
                    "false_alarms": ev_te["false_alarms"],
                    "early_min": ev_te["early_min"],
                    "peak_windows": ev_te["peak_windows"],
                    "params": "",
                    "_weights": weights,
                    "_pred_test": pred_te,
                })
                bar.update(1)
    bar.close()
    return rows


# 최종 앙상블 선택: 기본 3개 조합 중 valid 1등, 4개가 2% 이상 좋으면 4개
def select_ensemble(ens_rows):
    best3 = None
    best4 = None
    best_any = None

    for row in ens_rows:
        if best_any is None or row["valid_mae"] < best_any["valid_mae"]:
            best_any = row
        if row["n_models"] == 3:
            if best3 is None or row["valid_mae"] < best3["valid_mae"]:
                best3 = row
        if row["n_models"] == 4:
            if best4 is None or row["valid_mae"] < best4["valid_mae"]:
                best4 = row

    if best3 is None:
        return best_any            # 모델이 3개 미만일 때
    if best4 is not None and best4["valid_mae"] <= best3["valid_mae"] * (1 - SWITCH_GAIN):
        return best4
    return best3


# CSV 에 이어 쓰기 (처음이면 제목줄 포함)
def append_csv(rows, path):
    if len(rows) == 0:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def public_row(row):
    clean = {}
    for key, value in row.items():
        if not key.startswith("_"):
            clean[key] = value
    return clean


# ---------------------------------------------------------------
# 메인
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--include-db", action="store_true")
    args = parser.parse_args()

    names = args.models.split(",")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    global LOG_PATH
    LOG_PATH = os.path.join(OUT_DIR, "logs", f"train_{run_id}.log")
    log_write_print(f"=== 학습 시작 run_id={run_id} ===")
    log_write_print(f"모델: {', '.join(names)} | Optuna {args.trials}회 | DB 포함: {args.include_db}")

    # 1. 데이터
    df, X_seq, scale = load_data(args.include_db)
    df["split"] = features.split_by_date(df)
    df["residual"] = df["actual_kw"] - df["proj_base"]

    parts = {}
    data = {}
    for split in ["train", "valid", "test"]:
        mask = (df["split"] == split).to_numpy()
        parts[split] = df[mask].reset_index(drop=True)
        data[split] = (
            parts[split][TAB_COLS].to_numpy(dtype="float32"),
            X_seq[mask],
            parts[split]["residual"].to_numpy(dtype="float64"),
        )
        n_days = parts[split]["date"].nunique()
        log_write_print(f"  {split}: {mask.sum():,}개 샘플 / {n_days}일")

    # 평가용 목표전력: 학습 구간 15분 평균의 상위 5%
    train_windows = parts["train"].drop_duplicates("window_start")["actual_kw"]
    target_kw = float(np.percentile(train_windows, TARGET_PCT))
    log_write_print(f"  평가 목표전력: {target_kw:,.0f} kW (학습 구간 15분 평균 상위 {100 - TARGET_PCT}%)")

    result_rows = []
    preds = {}
    valid_mae = {}
    fitted = {}
    params_of = {}

    # 2. 기준선 (학습 없음)
    baselines = {"base_formula": "proj_base", "base_trend": "proj_trend"}
    for label, col in baselines.items():
        ev_va = evaluate(parts["valid"], parts["valid"][col].to_numpy(), target_kw, with_peaks=False)
        ev_te = evaluate(parts["test"], parts["test"][col].to_numpy(), target_kw, with_peaks=True)
        result_rows.append({
            "run_id": run_id, "kind": "baseline", "method": "", "n_models": 0,
            "members": label, "weights": "",
            "valid_mae": ev_va["mae"], "test_mae": ev_te["mae"], "test_mae_early": ev_te["mae_early"],
            "missed_peaks": ev_te["missed_peaks"], "false_alarms": ev_te["false_alarms"],
            "early_min": ev_te["early_min"], "peak_windows": ev_te["peak_windows"],
            "params": "", "selected": "",
        })
        log_write_print(f"[기준] {label:12s} {result_text(result_rows[-1])}")

    # 3. 단일 모델 6개
    for name in names:
        log_write_print()
        log_write_print(f"[학습] {name} 튜닝 중...")
        model, best = tune_and_fit(name, data, args.trials)
        fitted[name] = model
        params_of[name] = best

        preds[name] = {}
        for split in ["valid", "test"]:
            X, S, _ = data[split]
            preds[name][split] = parts[split]["proj_base"].to_numpy() + model.predict(X, S)

        ev_va = evaluate(parts["valid"], preds[name]["valid"], target_kw, with_peaks=False)
        ev_te = evaluate(parts["test"], preds[name]["test"], target_kw, with_peaks=True)
        valid_mae[name] = ev_va["mae"]

        result_rows.append({
            "run_id": run_id, "kind": "single", "method": "", "n_models": 1,
            "members": name, "weights": "",
            "valid_mae": ev_va["mae"], "test_mae": ev_te["mae"], "test_mae_early": ev_te["mae_early"],
            "missed_peaks": ev_te["missed_peaks"], "false_alarms": ev_te["false_alarms"],
            "early_min": ev_te["early_min"], "peak_windows": ev_te["peak_windows"],
            "params": json.dumps(best), "selected": "",
        })
        log_write_print(f"[학습] {name:12s} {result_text(result_rows[-1])}")
        log_write_print(f"       파라미터: {json.dumps(best)}")

    # 4. 앙상블 전 조합
    if len(names) >= 2:
        log_write_print()
        log_write_print("[앙상블] 전 조합 평가 중...")
        ens_rows = make_ensembles(names, preds, valid_mae, parts, target_kw, run_id)
        chosen = select_ensemble(ens_rows)
    else:
        ens_rows = []
        chosen = None

    ens_public = []
    for row in ens_rows:
        r = public_row(row)
        r["selected"] = "Y" if row is chosen else ""
        ens_public.append(r)
    ranked = sorted(ens_public, key=lambda r: r["valid_mae"])

    # 앙상블 전 조합 결과는 로그에만 기록 (valid 오차 순)
    for i, r in enumerate(ranked, start=1):
        mark = "  <- 선택" if r["selected"] == "Y" else ""
        log_write_print(f"  {i:3d}. {r['method']:8s} {r['members']:28s} 가중치 {r['weights']:24s} "
                        f"{result_text(r)}{mark}")

    # 5. CSV 저장: results.csv 에는 기준 + 단일 모델 + 앙상블 valid 상위 5개 + 선택된 것
    top = ranked[:5]
    for r in ranked:
        if r["selected"] == "Y" and r not in top:
            top.append(r)
    append_csv(result_rows + top, f"{OUT_DIR}/results.csv")

    # 6. 모델 저장 (앙상블, 모델이 1개면 그 모델 하나)
    if chosen is not None:
        members = chosen["members"].split("+")
        weights = chosen["_weights"]
        new_test_mae = chosen["test_mae"]
    else:
        members = names
        weights = [1.0]
        new_test_mae = result_rows[-1]["test_mae"]

    bundle = {
        "run_id": run_id,
        "members": members,
        "weights": weights,
        "models": {m: fitted[m] for m in members},
        "params": {m: params_of[m] for m in members},
        "target_kw": target_kw,
        "scale": scale,
        "test_mae": new_test_mae,
    }

    # 기존 모델이 있으면 같은 시험 데이터로 비교해서 더 좋을 때만 교체
    old_test_mae = None
    if os.path.exists(MODEL_PATH):
        old = joblib.load(MODEL_PATH)
        old_pred = predict_bundle(old, parts["test"], data["test"][1])
        old_test_mae = float(np.mean(np.abs(old_pred - parts["test"]["actual_kw"].to_numpy())))

    promoted = old_test_mae is None or new_test_mae <= old_test_mae
    if promoted:
        joblib.dump(bundle, MODEL_PATH)

    append_csv([{
        "run_id": run_id,
        "include_db": args.include_db,
        "new_model": "+".join(members),
        "new_test_mae": new_test_mae,
        "old_test_mae": old_test_mae,
        "promoted": "Y" if promoted else "N",
    }], f"{OUT_DIR}/retrain_log.csv")

    log_write_print()
    log_write_print(f"선택: {'+'.join(members)} (가중치 {[round(w, 2) for w in weights]})")
    for m in members:
        log_write_print(f"  {m} 파라미터: {json.dumps(params_of[m])}")
    log_write_print(f"시험 오차: 새 모델 {new_test_mae:.1f} kW | 기존 모델 {old_test_mae if old_test_mae is None else round(old_test_mae, 1)}")
    log_write_print("model.pkl 교체" if promoted else "기존 model.pkl 유지")
    log_write_print(f"로그 저장: {LOG_PATH}")


if __name__ == "__main__":
    main()
