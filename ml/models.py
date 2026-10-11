# 모델 6개 정의 + 앙상블 예측
#
# 모든 모델은 같은 사용법:
#   model = make_model(이름, 파라미터)
#   model.fit(X_tab, X_seq, y)        # y = 실제 평균 - 기존 계산식 (보정값)
#   model.predict(X_tab, X_seq)       # 보정값 예측 (kW)
#
# 최종 예측 = 기존 계산식 + 보정값
# 모델이 아무것도 못 배워도 보정값 0 = 기존 계산식이라 크게 나빠지지 않음

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import NORM, SEQ_EXTRA_COLS, TAB_COLS

MODEL_NAMES = ["ridge", "rf", "lgbm", "xgb", "gru", "cnn"]
SEQ_MODELS = ["gru", "cnn"]
EXTRA_IDX = [TAB_COLS.index(c) for c in SEQ_EXTRA_COLS]
BATCH = 512


# ---------------------------------------------------------------
# Optuna 탐색 범위
# ---------------------------------------------------------------
def suggest_params(name, trial):
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        }

    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if name == "gru":
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [3, 5]),
        }

    if name == "cnn":
        return {
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [3, 5]),
        }

    raise ValueError(f"모르는 모델: {name}")


# ---------------------------------------------------------------
# 트리·선형 모델 (요약 특징만 사용)
# ---------------------------------------------------------------
class TabModel:

    def __init__(self, name, params):
        self.name = name
        self.params = params

        if name == "ridge":
            self.model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=params["alpha"]))])
        elif name == "rf":
            self.model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        elif name == "lgbm":
            self.model = LGBMRegressor(**params, subsample_freq=1, verbose=-1, random_state=42)
        elif name == "xgb":
            self.model = XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"모르는 모델: {name}")

    def fit(self, X_tab, X_seq, y):
        self.model.fit(X_tab, y)

    def predict(self, X_tab, X_seq):
        return self.model.predict(X_tab)


# ---------------------------------------------------------------
# 딥러닝 모델 (최근 10분 시퀀스 + 보조 입력)
# ---------------------------------------------------------------
class GRUNet(nn.Module):

    def __init__(self, hidden, n_extra):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_extra, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, extra):
        out, _ = self.gru(seq.unsqueeze(-1))     # (배치, 60, 1) -> (배치, 60, hidden)
        last = out[:, -1]                        # 마지막 시점의 상태
        return self.head(torch.cat([last, extra], dim=1)).squeeze(1)


class CNNNet(nn.Module):

    def __init__(self, channels, n_extra):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(channels + n_extra, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, extra):
        feat = self.conv(seq.unsqueeze(1)).squeeze(-1)   # (배치, 1, 60) -> (배치, channels)
        return self.head(torch.cat([feat, extra], dim=1)).squeeze(1)


class SeqModel:

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.net = None

    # 보조 입력: kW 단위 값은 NORM 으로 나눠 크기를 맞춤
    def make_extra(self, X_tab):
        extra = X_tab[:, EXTRA_IDX].astype("float32").copy()
        for j, col in enumerate(SEQ_EXTRA_COLS):
            if col in ("proj_base", "kw_now"):
                extra[:, j] = extra[:, j] / NORM
        return extra

    def build_net(self):
        n_extra = len(SEQ_EXTRA_COLS)
        if self.name == "gru":
            return GRUNet(self.params["hidden"], n_extra)
        return CNNNet(self.params["channels"], n_extra)

    def fit(self, X_tab, X_seq, y):
        torch.manual_seed(42)
        self.net = self.build_net()

        seq = torch.tensor(X_seq, dtype=torch.float32)
        extra = torch.tensor(self.make_extra(X_tab))
        target = torch.tensor(y / NORM, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.HuberLoss()
        n = len(target)

        for epoch in range(self.params["epochs"]):
            self.net.train()
            order = torch.randperm(n)
            for start in range(0, n, BATCH):
                idx = order[start:start + BATCH]
                optimizer.zero_grad()
                pred = self.net(seq[idx], extra[idx])
                loss = loss_fn(pred, target[idx])
                loss.backward()
                optimizer.step()

    def predict(self, X_tab, X_seq):
        self.net.eval()
        seq = torch.tensor(X_seq, dtype=torch.float32)
        extra = torch.tensor(self.make_extra(X_tab))
        outputs = []

        with torch.no_grad():
            for start in range(0, len(seq), BATCH):
                pred = self.net(seq[start:start + BATCH], extra[start:start + BATCH])
                outputs.append(pred.numpy())

        if len(outputs) == 0:
            return np.zeros(0)
        return np.concatenate(outputs) * NORM


def make_model(name, params):
    if name in SEQ_MODELS:
        return SeqModel(name, params)
    return TabModel(name, params)


# ---------------------------------------------------------------
# 저장된 앙상블(model.pkl)로 예측
#   bundle: train.py 가 저장한 dict (members, weights, models)
#   df    : TAB_COLS 를 가진 DataFrame
# ---------------------------------------------------------------
def predict_bundle(bundle, df, X_seq):
    X_tab = df[TAB_COLS].to_numpy(dtype="float32")
    base = df["proj_base"].to_numpy(dtype="float64")
    total = np.zeros(len(df))

    for name, weight in zip(bundle["members"], bundle["weights"]):
        residual = bundle["models"][name].predict(X_tab, X_seq)
        total = total + weight * (base + residual)

    return total
