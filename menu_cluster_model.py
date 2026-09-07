import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class MenuNutritionCluster:
    """
    메뉴 영양 프로파일을 비지도학습(K-Means)으로 군집화한다.

    - 탄수화물/단백질/지방: 총 에너지 대비 비율
    - 식이섬유/나트륨/무기질/비타민: 1000 kcal당 영양밀도
    - k=3~7 중 silhouette score가 가장 높은 k를 자동 선택
    """

    def __init__(self, k_min=3, k_max=7, random_state=42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        self.model = None
        self.best_k_ = None
        self.silhouette_score_ = None
        self.feature_names_ = None
        self.cluster_search_ = None

    def _build_features(self, df):
        energy = pd.to_numeric(df["에너지"], errors="coerce")
        valid_energy = energy.where(energy > 0)

        X = pd.DataFrame(index=df.index)

        # 다량영양소: 에너지 비율
        macro_specs = {
            "carb_energy_ratio": ("탄수화물", 4),
            "protein_energy_ratio": ("단백질", 4),
            "fat_energy_ratio": ("지방", 9),
        }

        for new_col, (src_col, kcal_per_g) in macro_specs.items():
            if src_col in df.columns:
                nutrient = pd.to_numeric(df[src_col], errors="coerce")
                X[new_col] = nutrient * kcal_per_g / valid_energy * 100

        # 미량/제한 영양소: 1000 kcal당 영양밀도
        density_cols = [
            "총 식이섬유",
            "나트륨",
            "칼슘",
            "철",
            "마그네슘",
            "칼륨",
            "아연",
            "비타민 A",
            "비타민 C",
            "비타민 D",
        ]

        for col in density_cols:
            if col in df.columns:
                nutrient = pd.to_numeric(df[col], errors="coerce")
                X[f"{col}_density"] = nutrient / valid_energy * 1000

        X = X.replace([np.inf, -np.inf], np.nan)

        # 전부 결측인 특성은 제외
        X = X.dropna(axis=1, how="all")

        if X.shape[1] < 2:
            raise ValueError("군집화에 사용할 유효 영양 특성이 부족합니다.")

        return X

    def fit(self, df):
        X = self._build_features(df)
        self.feature_names_ = X.columns.tolist()

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        search_rows = []
        best_score = -np.inf
        best_k = None

        max_k = min(self.k_max, len(df) - 1)

        for k in range(self.k_min, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = model.fit_predict(X_scaled)

            score = silhouette_score(
                X_scaled,
                labels,
                sample_size=min(1000, len(df)),
                random_state=self.random_state
            )

            search_rows.append({
                "k": k,
                "silhouette_score": score
            })

            if score > best_score:
                best_score = score
                best_k = k

        self.best_k_ = best_k
        self.silhouette_score_ = best_score
        self.cluster_search_ = pd.DataFrame(search_rows)

        self.model = KMeans(
            n_clusters=self.best_k_,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X_scaled)

        return self

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("fit()을 먼저 실행해야 합니다.")

        X = self._build_features(df)

        # 학습 당시 특성 순서 유지
        X = X.reindex(columns=self.feature_names_)

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return self.model.predict(X_scaled)

    def fit_predict(self, df):
        self.fit(df)

        result = df.copy()
        result["nutrition_cluster"] = self.predict(df)

        return result

    def cluster_summary(self, clustered_df):
        if "nutrition_cluster" not in clustered_df.columns:
            raise ValueError("nutrition_cluster 컬럼이 없습니다.")

        X = self._build_features(clustered_df)
        summary_df = X.copy()
        summary_df["nutrition_cluster"] = clustered_df["nutrition_cluster"].values

        summary = (
            summary_df
            .groupby("nutrition_cluster")
            .mean()
            .round(2)
        )

        summary.insert(
            0,
            "menu_count",
            clustered_df.groupby("nutrition_cluster").size()
        )

        return summary.reset_index()
