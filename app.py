from __future__ import annotations

import json
import os
import re
from itertools import product
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI

from menu_cluster_model import MenuNutritionCluster


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RAG_DIR = BASE_DIR / "rag_store"

MENU_PATH = DATA_DIR / "menugen_menu_master.csv"
INGREDIENT_PATH = DATA_DIR / "menugen_menu_ingredients.csv"
NUTRITION_PATH = DATA_DIR / "food_nutrition.xlsx"

EMBEDDING_PATH = RAG_DIR / "bge_m3_menu_embeddings.npy"
DOCUMENT_PATH = RAG_DIR / "menu_documents.csv"
FAISS_PATH = RAG_DIR / "menu_index.faiss"

BGE_MODEL_NAME = "BAAI/bge-m3"
OPENAI_MODEL = "gpt-5.6-luna"

PRE_GATE_POOL_SIZE = {
    "주식": 12,
    "국": 15,
    "주찬": 20,
    "부찬": 25,
    "김치": 10,
}

FINAL_POOL_SIZE = {
    "주식": 8,
    "국": 8,
    "주찬": 8,
    "부찬": 12,
    "김치": 8,
}

MEAL_SLOT_GROUPS = {
    "주식": {"밥류", "면 및 만두류", "죽류"},
    "국": {"국(탕)류", "찌개류"},
    "주찬": {"구이류", "볶음류", "조림류", "찜류", "전류", "튀김류"},
    "부찬": {"무침류", "볶음류", "조림류", "전류", "절임류"},
    "김치": {"김치류"},
}

MEAL_SLOTS = ["주식", "국", "주찬", "부찬", "김치"]

NON_MENU_GROUPS = {"원재료", "장류", "주류"}
EXCLUDED_MENU_NAMES = {"육수(소고기)"}

INFO_COLS = ["DB10.4 색인", "식품군", "식품명", "출처"]

# 노트북 V1에서 최종 선택한 영양 지표
LLM_FEATURE_COLS = [
    "DB10.4 색인", "식품군", "식품명", "출처",
    "에너지", "수분", "단백질", "지방", "탄수화물", "당류",
    "총 식이섬유", "비타민 A", "비타민 D", "비타민 C",
    "티아민", "리보플라빈", "비타민 B12", "비타민 K1",
    "칼슘", "철", "마그네슘", "인", "칼륨", "나트륨", "아연",
    "요오드", "총 필수 아미노산", "총 포화 지방산",
    "총 불포화 지방산", "오메가3 지방산", "오메가6 지방산",
    "총 트랜스 지방산", "콜레스테롤",
]


load_dotenv(BASE_DIR / ".env")
client = OpenAI()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY가 없습니다. 프로젝트 루트의 .env 파일에 "
        "OPENAI_API_KEY=... 를 설정하세요."
    )


# ============================================================
# 1. 데이터 전처리 / 통합
# ============================================================

def clean_column_names(columns):
    return (
        pd.Index(columns)
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_nutrition_value(value):
    """
    국가표준식품성분 DB 10.4 표기 처리
    - '-'   : 미측정 -> NaN
    - 'Tr'  : 미량 -> NaN
    - '(수치)' : 인용/환산값 -> 수치
    - '(Tr)' : 인용/환산 미량 -> NaN
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "-":
        return np.nan

    if value.lower() in {"tr", "(tr)"}:
        return np.nan

    if re.fullmatch(r"\([\d,]+(?:\.\d+)?\)", value):
        return float(value[1:-1].replace(",", ""))

    numeric_value = pd.to_numeric(
        value.replace(",", ""),
        errors="coerce",
    )

    return float(numeric_value) if pd.notna(numeric_value) else np.nan


def normalize_food_name(name):
    if pd.isna(name):
        return ""

    name = str(name).strip()
    name = name.replace("，", ",")
    name = re.sub(r"\s+", "", name)
    return name.casefold()


def join_unique_ingredients(values):
    return ", ".join(
        dict.fromkeys(
            str(v).strip()
            for v in values
            if pd.notna(v) and str(v).strip()
        )
    )


def safe_text(value, default="없음"):
    if pd.isna(value):
        return default

    value = str(value).strip()
    return value if value else default


def make_menu_document(row):
    return "\n".join([
        f"메뉴명: {row['menu_fd_Nm']}",
        f"메뉴 대분류: {safe_text(row['upper_Fd_Grupp_Nm'])}",
        f"메뉴 소분류: {safe_text(row['fd_Grupp_Nm'])}",
        f"식재료: {safe_text(row['ingredient_text'])}",
    ])


def load_service_data():
    required_files = [MENU_PATH, INGREDIENT_PATH, NUTRITION_PATH]
    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "필수 데이터 파일이 없습니다:\n" + "\n".join(missing)
        )

    menu_df = pd.read_csv(MENU_PATH)
    ingredient_df = pd.read_csv(INGREDIENT_PATH)

    # 코드 타입 차이 방지
    menu_df["fd_Code"] = menu_df["fd_Code"].astype(str)
    ingredient_df["menu_fd_Code"] = ingredient_df["menu_fd_Code"].astype(str)

    # 국가표준식품성분 DB10.4의 실제 헤더 구성
    nutrition_raw = pd.read_excel(
        NUTRITION_PATH,
        sheet_name="국가표준식품성분 Database 10.4",
        header=None,
    )

    upper_header = nutrition_raw.iloc[0]
    detail_header = nutrition_raw.iloc[1]

    column_names = detail_header.where(
        detail_header.notna(),
        upper_header,
    )
    column_names = clean_column_names(column_names)

    nutrition_df = (
        nutrition_raw
        .iloc[3:]
        .reset_index(drop=True)
    )
    nutrition_df.columns = column_names

    # 서비스에서 실제 사용하는 컬럼만 선택
    final_cols = [
        col
        for col in LLM_FEATURE_COLS
        if col in nutrition_df.columns
    ]

    llm_nutrition_df = nutrition_df[final_cols].copy()

    nutrient_cols = [
        col
        for col in final_cols
        if col not in INFO_COLS
    ]

    # 숫자형 영양값 파싱
    for col in nutrient_cols:
        llm_nutrition_df[col] = (
            llm_nutrition_df[col]
            .map(parse_nutrition_value)
        )

    # 식품명 정규화
    ingredient_df["food_name_norm"] = (
        ingredient_df["ingredient_food_Nm"]
        .map(normalize_food_name)
    )

    llm_nutrition_df["food_name_norm"] = (
        llm_nutrition_df["식품명"]
        .map(normalize_food_name)
    )

    # MenuGen 식재료 ↔ 국가표준식품성분DB 연결
    ingredient_nutrition_df = ingredient_df.merge(
        llm_nutrition_df[
            ["food_name_norm"] + nutrient_cols
        ],
        on="food_name_norm",
        how="left",
    )

    # 100g 기준 영양값 -> 메뉴 실제 식재료 중량 기준 환산
    for col in nutrient_cols:
        ingredient_nutrition_df[col] = (
            ingredient_nutrition_df[col]
            * ingredient_nutrition_df["ingredient_food_Wgh"]
            / 100
        )

    # 메뉴별 영양성분 합산
    menu_nutrition_df = (
        ingredient_nutrition_df
        .groupby(
            ["menu_fd_Code", "menu_fd_Nm"],
            as_index=False,
        )[nutrient_cols]
        .sum(min_count=1)
    )

    # 메뉴별 알레르기 정보 집계
    menu_allergy_df = (
        ingredient_df
        .groupby(
            ["menu_fd_Code", "menu_fd_Nm"],
            as_index=False,
        )["ingredient_allrgy_Info"]
        .agg(
            lambda x: ", ".join(
                sorted(
                    set(
                        str(v).strip()
                        for v in x
                        if pd.notna(v) and str(v).strip()
                    )
                )
            )
        )
        .rename(
            columns={
                "ingredient_allrgy_Info": "allergy_info"
            }
        )
    )

    # 메뉴 영양 + 알레르기
    menu_rag_df = menu_nutrition_df.merge(
        menu_allergy_df,
        on=["menu_fd_Code", "menu_fd_Nm"],
        how="left",
    )

    # 메뉴 대/소분류
    menu_meta_df = menu_df[
        ["fd_Code", "upper_Fd_Grupp_Nm", "fd_Grupp_Nm"]
    ].copy()

    menu_meta_df = menu_meta_df.rename(
        columns={"fd_Code": "menu_fd_Code"}
    )

    menu_rag_df = menu_rag_df.merge(
        menu_meta_df,
        on="menu_fd_Code",
        how="left",
    )

    # 실제 추천 대상 메뉴만 유지
    recommendable_menu_df = (
        menu_rag_df[
            ~menu_rag_df["upper_Fd_Grupp_Nm"].isin(NON_MENU_GROUPS)
            &
            ~menu_rag_df["menu_fd_Nm"].isin(EXCLUDED_MENU_NAMES)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # 메뉴별 식재료 텍스트
    ingredient_text_map = (
        ingredient_df
        .groupby("menu_fd_Code")["ingredient_food_Nm"]
        .agg(join_unique_ingredients)
        .to_dict()
    )

    recommendable_menu_df["ingredient_text"] = (
        recommendable_menu_df["menu_fd_Code"]
        .map(ingredient_text_map)
        .fillna("")
    )

    menu_documents_df = recommendable_menu_df[
        [
            "menu_fd_Code",
            "menu_fd_Nm",
            "upper_Fd_Grupp_Nm",
            "fd_Grupp_Nm",
            "ingredient_text",
        ]
    ].copy()

    menu_documents_df["document"] = (
        recommendable_menu_df.apply(
            make_menu_document,
            axis=1,
        )
    )

    return (
        recommendable_menu_df,
        menu_documents_df,
        nutrient_cols,
    )


# ============================================================
# 2. BGE-M3 / FAISS 로드
# ============================================================

def align_to_saved_documents(
    recommendable_df,
    generated_documents_df,
):
    """
    기존 notebook에서 저장한 menu_documents.csv가 있으면
    그 순서에 맞춰 DataFrame을 정렬하여 저장 embedding과 행을 일치시킨다.
    """
    if not DOCUMENT_PATH.exists():
        return recommendable_df, generated_documents_df

    saved_docs = pd.read_csv(DOCUMENT_PATH)
    saved_docs["menu_fd_Code"] = (
        saved_docs["menu_fd_Code"].astype(str)
    )

    recommendable_df = recommendable_df.copy()
    recommendable_df["menu_fd_Code"] = (
        recommendable_df["menu_fd_Code"].astype(str)
    )

    saved_codes = saved_docs["menu_fd_Code"].tolist()
    current_codes = recommendable_df["menu_fd_Code"].tolist()

    if (
        len(saved_codes) == len(current_codes)
        and set(saved_codes) == set(current_codes)
    ):
        order_df = pd.DataFrame({
            "menu_fd_Code": saved_codes,
            "_order": range(len(saved_codes)),
        })

        recommendable_df = (
            order_df
            .merge(
                recommendable_df,
                on="menu_fd_Code",
                how="left",
            )
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        generated_documents_df = saved_docs.copy()

    return recommendable_df, generated_documents_df


def load_or_build_embeddings(
    bge_model,
    menu_documents_df,
):
    RAG_DIR.mkdir(exist_ok=True)

    if EMBEDDING_PATH.exists():
        embeddings = np.load(EMBEDDING_PATH)

        if len(embeddings) == len(menu_documents_df):
            print(
                f"[RAG] 저장 embedding 사용: {embeddings.shape}"
            )
            return embeddings.astype("float32")

        print(
            "[RAG] 저장 embedding 행 수 불일치 -> 재생성"
        )

    # rag_store가 없을 때만 최초 1회 생성
    print(
        "[RAG] embedding 파일이 없어 BGE-M3 embedding을 생성합니다."
    )

    menu_texts = menu_documents_df["document"].tolist()

    output = bge_model.encode(
        menu_texts,
        batch_size=8,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    embeddings = np.asarray(
        output["dense_vecs"],
        dtype="float32",
    )

    np.save(
        EMBEDDING_PATH,
        embeddings,
    )

    menu_documents_df.to_csv(
        DOCUMENT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return embeddings


def build_slot_indexes(
    recommendable_df,
    embeddings,
):
    slot_indexes = {}
    slot_menu_dfs = {}

    for slot_name, allowed_groups in MEAL_SLOT_GROUPS.items():
        mask = (
            recommendable_df["upper_Fd_Grupp_Nm"]
            .isin(allowed_groups)
            .to_numpy()
        )

        slot_df = (
            recommendable_df[mask]
            .copy()
            .reset_index(drop=True)
        )

        slot_vectors = embeddings[mask].copy()
        faiss.normalize_L2(slot_vectors)

        slot_index = faiss.IndexFlatIP(
            slot_vectors.shape[1]
        )
        slot_index.add(slot_vectors)

        slot_menu_dfs[slot_name] = slot_df
        slot_indexes[slot_name] = slot_index

    return slot_menu_dfs, slot_indexes


# ============================================================
# 3. LLM 조건 추출 / 역할 Gate
# ============================================================

def extract_user_conditions(user_query):
    response = client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        input=f"""
너는 급식 메뉴 추천 시스템의 조건 추출기다.

사용자의 요청에서 실제 메뉴 검색과 영양 랭킹에 필요한 조건만 추출해라.

규칙:
- target: 급식 대상
- allergies: 알레르기 목록
- nutrition_high: 많이 섭취하고 싶은 영양소
- nutrition_low: 적게 섭취하고 싶은 영양소
- diseases: 질환 또는 건강 상태
- keywords: 위 항목에 포함되지 않는 실제 추천 조건

'급식', '메뉴', '추천', '음식'처럼
추천 시스템 자체를 설명하는 일반적인 단어는 keywords에 넣지 마라.

사용자 요청:
{user_query}
""",
        text={
            "format": {
                "type": "json_schema",
                "name": "meal_conditions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": ["string", "null"]
                        },
                        "allergies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "nutrition_high": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "nutrition_low": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "diseases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "target",
                        "allergies",
                        "nutrition_high",
                        "nutrition_low",
                        "diseases",
                        "keywords",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(response.output_text)


def evaluate_slot_candidates(
    slot_gate_items,
    conditions,
):
    target_context = " ".join([
        str(conditions.get("target") or ""),
        *conditions.get("keywords", []),
    ]).strip()

    response = client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        input=f"""
너는 급식 식단의 메뉴 역할 적합성을 판정하는 분류기다.

급식 대상:
{target_context}

각 후보가 현재 지정된 슬롯에 실제 급식 식단 구성상
자연스러운 경우에만 keep=true로 판단한다.

[슬롯 기준]

주식:
- 밥, 죽, 면 등 식사의 중심 탄수화물 음식
- 이유식처럼 현재 급식 대상과 명백히 맞지 않으면 제외

국:
- 국, 탕, 찌개 등 국물 음식
- 메뉴명이 국/탕류로 분류되었더라도
  통닭·백숙 등 독립적인 주찬 성격이 강하면 제외

주찬:
- 한 끼의 중심 반찬
- 육류, 생선, 달걀, 두부 등 단백질 중심 또는
  일반적으로 메인 반찬으로 제공되는 음식
- 간식, 디저트, 떡류, 일품식,
  채소 단독 소량 반찬은 제외

부찬:
- 주찬을 보조하는 일반적인 반찬
- 나물, 무침, 조림, 채소반찬 등
- 떡볶이·면·밥처럼 주식성 또는 일품식 성격이 강한 음식,
  간식·디저트는 제외

김치:
- 김치류만 유지

추가 규칙:
- 영양성분을 보고 판단하지 않는다.
- 질환 치료 적합성을 판단하지 않는다.
- 메뉴명과 분류를 근거로 식단 역할만 판단한다.
- 명백히 부자연스러운 슬롯 배치는 제외한다.
- 합리적으로 해당 역할에 제공될 수 있다면 유지한다.
- 메뉴명에 어린이, 유아, 영유아 등 특정 대상이 명시되어 있고
  현재 급식 대상과 명백히 다르면 제외한다.

검사 대상:
{json.dumps(
    slot_gate_items,
    ensure_ascii=False,
    indent=2,
)}
""",
        text={
            "format": {
                "type": "json_schema",
                "name": "slot_candidate_gate",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "candidate_id": {
                                        "type": "string"
                                    },
                                    "keep": {
                                        "type": "boolean"
                                    },
                                    "reason": {
                                        "type": "string"
                                    },
                                },
                                "required": [
                                    "candidate_id",
                                    "keep",
                                    "reason",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(response.output_text)


# ============================================================
# 4. RAG 검색 / Allergy Hard Filter
# ============================================================

def filter_allergies(df, allergies):
    result = df.copy()

    for allergy in allergies:
        mask = (
            result["allergy_info"]
            .fillna("")
            .str.contains(
                str(allergy),
                case=False,
                regex=False,
            )
        )
        result = result[~mask]

    return result


def retrieve_slot_candidates(
    slot_name,
    semantic_query="",
    allergies=None,
    top_k=15,
):
    if allergies is None:
        allergies = []

    slot_df = SLOT_MENU_DFS[slot_name]
    slot_index = SLOT_INDEXES[slot_name]

    query = f"{slot_name} 급식 메뉴"

    if semantic_query:
        query += "\n추가 조건: " + semantic_query

    query_output = BGE_MODEL.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query_vector = np.asarray(
        query_output["dense_vecs"],
        dtype="float32",
    )

    faiss.normalize_L2(query_vector)

    # 슬롯 전체 검색 후 알레르기 필터 -> 상위 N
    search_k = slot_index.ntotal

    scores, indices = slot_index.search(
        query_vector,
        search_k,
    )

    retrieved = (
        slot_df
        .iloc[indices[0]]
        .copy()
    )

    retrieved["retrieval_score"] = scores[0]

    retrieved = filter_allergies(
        retrieved,
        allergies,
    )

    return (
        retrieved
        .head(top_k)
        .reset_index(drop=True)
    )


def build_slot_gate_items(slot_candidates):
    items = []

    for slot_name, df in slot_candidates.items():
        for idx, row in df.iterrows():
            items.append({
                "candidate_id": f"{slot_name}_{idx}",
                "slot": slot_name,
                "menu_name": row["menu_fd_Nm"],
                "upper_group": row["upper_Fd_Grupp_Nm"],
                "sub_group": row["fd_Grupp_Nm"],
            })

    return items


def apply_slot_gate(
    slot_candidates,
    slot_gate_result,
):
    # 누락 응답이 있으면 조용히 통과시키지 않고 실패시킴
    expected_ids = {
        f"{slot_name}_{idx}"
        for slot_name, df in slot_candidates.items()
        for idx in range(len(df))
    }

    returned_ids = {
        item["candidate_id"]
        for item in slot_gate_result["results"]
    }

    if expected_ids != returned_ids:
        raise ValueError(
            "Meal-role Gate 응답의 후보 ID가 입력 후보와 일치하지 않습니다."
        )

    keep_map = {
        item["candidate_id"]: item["keep"]
        for item in slot_gate_result["results"]
    }

    filtered_slots = {}

    for slot_name, df in slot_candidates.items():
        keep_indices = []

        for idx in range(len(df)):
            candidate_id = f"{slot_name}_{idx}"

            if keep_map[candidate_id]:
                keep_indices.append(idx)

        filtered_slots[slot_name] = (
            df.iloc[keep_indices]
            .copy()
            .reset_index(drop=True)
        )

    return filtered_slots


# ============================================================
# 5. 식단 Combination
# ============================================================

def build_meal_combinations(slot_candidates):
    combinations = []

    for staple, soup, main, side, kimchi in product(
        slot_candidates["주식"].to_dict("records"),
        slot_candidates["국"].to_dict("records"),
        slot_candidates["주찬"].to_dict("records"),
        slot_candidates["부찬"].to_dict("records"),
        slot_candidates["김치"].to_dict("records"),
    ):
        menus = {
            "주식": staple,
            "국": soup,
            "주찬": main,
            "부찬": side,
            "김치": kimchi,
        }

        # 같은 메뉴가 두 슬롯에 동시에 들어가는 조합 제거
        menu_codes = [
            menu["menu_fd_Code"]
            for menu in menus.values()
        ]

        if len(menu_codes) != len(set(menu_codes)):
            continue

        combination = {
            "주식": staple["menu_fd_Nm"],
            "국": soup["menu_fd_Nm"],
            "주찬": main["menu_fd_Nm"],
            "부찬": side["menu_fd_Nm"],
            "김치": kimchi["menu_fd_Nm"],
        }

        for slot_name, menu in menus.items():
            combination[f"{slot_name}_code"] = (
                menu["menu_fd_Code"]
            )

        # 식단 전체 영양성분 합산
        for nutrient in NUTRIENT_COLS:
            values = [
                menu.get(nutrient)
                for menu in menus.values()
            ]

            valid_values = [
                value
                for value in values
                if pd.notna(value)
            ]

            combination[nutrient] = (
                sum(valid_values)
                if valid_values
                else np.nan
            )

        combination["retrieval_score"] = np.mean([
            menu["retrieval_score"]
            for menu in menus.values()
        ])

        combinations.append(combination)

    return pd.DataFrame(combinations)


# ============================================================
# 6. Multi-Nutrient Ranking
# ============================================================

def range_score(series, lower, upper):
    score = pd.Series(
        1.0,
        index=series.index,
    )

    width = upper - lower

    low_mask = series < lower
    high_mask = series > upper

    score.loc[low_mask] = (
        1
        - (lower - series.loc[low_mask])
        / width
    )

    score.loc[high_mask] = (
        1
        - (series.loc[high_mask] - upper)
        / width
    )

    return score.clip(
        lower=0,
        upper=1,
    )


def rank_meal_combinations(
    meal_df,
    nutrition_high,
    nutrition_low,
    top_k=2000,
):
    result = meal_df.copy()

    # 1) 다량영양소 에너지 비율
    result = result[
        result["에너지"] > 0
    ].copy()

    result["carb_energy_ratio"] = (
        result["탄수화물"]
        * 4
        / result["에너지"]
        * 100
    )

    result["protein_energy_ratio"] = (
        result["단백질"]
        * 4
        / result["에너지"]
        * 100
    )

    result["fat_energy_ratio"] = (
        result["지방"]
        * 9
        / result["에너지"]
        * 100
    )

    # 2025 KDRI 에너지 적정비율
    result["carb_balance_score"] = range_score(
        result["carb_energy_ratio"],
        50,
        65,
    )

    result["protein_balance_score"] = range_score(
        result["protein_energy_ratio"],
        10,
        20,
    )

    result["fat_balance_score"] = range_score(
        result["fat_energy_ratio"],
        15,
        30,
    )

    result["macro_balance_score"] = (
        result[
            [
                "carb_balance_score",
                "protein_balance_score",
                "fat_balance_score",
            ]
        ]
        .mean(axis=1)
    )

    # 2) 미량영양소: 1000 kcal 기준 상대 영양밀도
    beneficial_cols = [
        "총 식이섬유",
        "칼슘",
        "철",
        "마그네슘",
        "칼륨",
        "아연",
        "비타민 A",
        "비타민 D",
        "비타민 C",
        "티아민",
        "리보플라빈",
        "비타민 B12",
        "비타민 K1",
        "오메가3 지방산",
    ]

    beneficial_scores = []

    for col in beneficial_cols:
        if col not in result.columns:
            continue

        density = (
            result[col]
            / result["에너지"]
            * 1000
        )

        if density.notna().any():
            score_col = f"_benefit_{col}"

            result[score_col] = (
                density
                .rank(pct=True)
                .fillna(0.5)
            )

            beneficial_scores.append(score_col)

    if beneficial_scores:
        result["micronutrient_score"] = (
            result[beneficial_scores]
            .mean(axis=1)
        )
    else:
        result["micronutrient_score"] = 0.5

    # 3) 제한 영양소
    limit_cols = [
        "당류",
        "나트륨",
        "총 포화 지방산",
        "총 트랜스 지방산",
        "콜레스테롤",
    ]

    limit_scores = []

    for col in limit_cols:
        if col not in result.columns:
            continue

        density = (
            result[col]
            / result["에너지"]
            * 1000
        )

        if density.notna().any():
            score_col = f"_limit_{col}"

            result[score_col] = (
                1 - density.rank(pct=True)
            ).fillna(0.5)

            limit_scores.append(score_col)

    if limit_scores:
        result["limit_nutrient_score"] = (
            result[limit_scores]
            .mean(axis=1)
        )
    else:
        result["limit_nutrient_score"] = 0.5

    # 4) 에너지 극단값 억제
    energy_pct = (
        result["에너지"]
        .rank(pct=True)
    )

    result["energy_center_score"] = (
        1
        - (energy_pct - 0.5).abs() * 2
    ).clip(
        lower=0,
        upper=1,
    )

    # 5) 기본 영양 품질
    result["baseline_nutrition_score"] = (
        result[
            [
                "macro_balance_score",
                "micronutrient_score",
                "limit_nutrient_score",
                "energy_center_score",
            ]
        ]
        .mean(axis=1)
    )

    # 6) 사용자 영양 요구
    preference_scores = []

    for col in nutrition_high:
        if col not in result.columns:
            continue

        score_col = f"_pref_high_{col}"

        if col == "단백질":
            result[score_col] = (
                result["protein_energy_ratio"]
                .clip(lower=0, upper=20)
                / 20
            )
        else:
            density = (
                result[col]
                / result["에너지"]
                * 1000
            )

            result[score_col] = (
                density
                .rank(pct=True)
                .fillna(0.5)
            )

        preference_scores.append(score_col)

    for col in nutrition_low:
        if col not in result.columns:
            continue

        score_col = f"_pref_low_{col}"

        density = (
            result[col]
            / result["에너지"]
            * 1000
        )

        result[score_col] = (
            1 - density.rank(pct=True)
        ).fillna(0.5)

        preference_scores.append(score_col)

    if preference_scores:
        result["preference_score"] = (
            result[preference_scores]
            .mean(axis=1)
        )

        result["nutrition_score"] = (
            result[
                [
                    "baseline_nutrition_score",
                    "preference_score",
                ]
            ]
            .mean(axis=1)
        )
    else:
        result["preference_score"] = 0.5
        result["nutrition_score"] = (
            result["baseline_nutrition_score"]
        )

    return (
        result
        .sort_values(
            ["nutrition_score", "retrieval_score"],
            ascending=[False, False],
        )
        .head(top_k)
        .reset_index(drop=True)
    )


# ============================================================
# 7. K-Means 영양 군집 + Diversity
# ============================================================

def add_ml_cluster_score(ranked_df):
    result = ranked_df.copy()
    cluster_cols = []

    for slot in MEAL_SLOTS:
        cluster_col = f"{slot}_cluster"

        result[cluster_col] = (
            result[f"{slot}_code"]
            .astype(str)
            .map(CLUSTER_MAP)
        )

        cluster_cols.append(cluster_col)

    result["cluster_diversity_score"] = (
        result[cluster_cols]
        .nunique(axis=1)
        / len(MEAL_SLOTS)
    )

    # 기존 영양랭킹을 중심으로 유지하고
    # K-Means cluster diversity를 보조적으로 반영
    result["final_score"] = (
        0.95 * result["nutrition_score"]
        + 0.05 * result["cluster_diversity_score"]
    )

    return (
        result
        .sort_values(
            [
                "final_score",
                "nutrition_score",
                "retrieval_score",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def select_diverse_meals(
    ranked_df,
    n_meals=3,
    max_shared_slots=1,
):
    selected = []

    for _, candidate in ranked_df.iterrows():
        if not selected:
            selected.append(candidate)
            continue

        is_diverse = True

        for chosen in selected:
            shared_slots = sum(
                candidate[slot] == chosen[slot]
                for slot in MEAL_SLOTS
            )

            if shared_slots > max_shared_slots:
                is_diverse = False
                break

        if is_diverse:
            selected.append(candidate)

        if len(selected) >= n_meals:
            break

    return (
        pd.DataFrame(selected)
        .reset_index(drop=True)
    )


# ============================================================
# 8. 추천 Service
# ============================================================

def make_semantic_query(conditions):
    semantic_parts = []

    if conditions.get("target"):
        semantic_parts.append(
            str(conditions["target"])
        )

    semantic_parts.extend(
        conditions.get("keywords", [])
    )

    return " ".join(semantic_parts)


def recommend_meals(user_query):
    if not user_query or not user_query.strip():
        raise gr.Error("추천 조건을 입력해주세요.")

    # 1. 자연어 조건 구조화
    conditions = extract_user_conditions(
        user_query.strip()
    )

    semantic_query = make_semantic_query(
        conditions
    )

    # 2. 슬롯별 Menu RAG
    slot_candidates = {}

    for slot_name in MEAL_SLOT_GROUPS:
        slot_candidates[slot_name] = (
            retrieve_slot_candidates(
                slot_name=slot_name,
                semantic_query=semantic_query,
                allergies=conditions.get(
                    "allergies",
                    [],
                ),
                top_k=PRE_GATE_POOL_SIZE[
                    slot_name
                ],
            )
        )

    # 3. LLM Target / Meal-role Gate
    gate_items = build_slot_gate_items(
        slot_candidates
    )

    gate_result = evaluate_slot_candidates(
        gate_items,
        conditions,
    )

    filtered_slots = apply_slot_gate(
        slot_candidates,
        gate_result,
    )

    # 4. 최종 Pool 제한
    for slot_name in filtered_slots:
        filtered_slots[slot_name] = (
            filtered_slots[slot_name]
            .head(FINAL_POOL_SIZE[slot_name])
            .reset_index(drop=True)
        )

    empty_slots = [
        slot_name
        for slot_name, df in filtered_slots.items()
        if df.empty
    ]

    if empty_slots:
        raise gr.Error(
            "후보가 없는 슬롯: "
            + ", ".join(empty_slots)
            + "\n조건을 조금 완화해 주세요."
        )

    # 5. Meal Combination
    meal_df = build_meal_combinations(
        filtered_slots
    )

    if meal_df.empty:
        raise gr.Error(
            "현재 조건에서 생성 가능한 식단 조합이 없습니다."
        )

    # 6. Multi-Nutrient Ranking
    ranked_df = rank_meal_combinations(
        meal_df,
        nutrition_high=conditions.get(
            "nutrition_high",
            [],
        ),
        nutrition_low=conditions.get(
            "nutrition_low",
            [],
        ),
        top_k=2000,
    )

    # 7. K-Means ML Cluster Diversity
    ranked_df = add_ml_cluster_score(
        ranked_df
    )

    # 8. 최종 Diversity Selection
    final_df = select_diverse_meals(
        ranked_df,
        n_meals=3,
        max_shared_slots=1,
    )

    # 화면용 결과
    result_columns = [
        "주식",
        "국",
        "주찬",
        "부찬",
        "김치",
        "에너지",
        "단백질",
        "나트륨",
        "nutrition_score",
        "cluster_diversity_score",
        "final_score",
        "retrieval_score",
    ]

    result_df = final_df[
        [
            col
            for col in result_columns
            if col in final_df.columns
        ]
    ].copy()

    rename_map = {
        "에너지": "에너지(kcal)",
        "단백질": "단백질(g)",
        "나트륨": "나트륨(mg)",
        "nutrition_score": "영양점수",
        "cluster_diversity_score": "ML군집다양성",
        "final_score": "최종점수",
        "retrieval_score": "검색유사도",
    }

    result_df = result_df.rename(
        columns=rename_map
    )

    for col in [
        "에너지(kcal)",
        "단백질(g)",
        "나트륨(mg)",
    ]:
        if col in result_df.columns:
            result_df[col] = (
                pd.to_numeric(
                    result_df[col],
                    errors="coerce",
                )
                .round(1)
            )

    for col in [
        "영양점수",
        "ML군집다양성",
        "최종점수",
        "검색유사도",
    ]:
        if col in result_df.columns:
            result_df[col] = (
                pd.to_numeric(
                    result_df[col],
                    errors="coerce",
                )
                .round(3)
            )

    condition_md = f"""
### 입력 조건 분석

- **급식 대상:** {conditions.get("target") or "미지정"}
- **알레르기:** {", ".join(conditions.get("allergies", [])) or "없음"}
- **강화 영양소:** {", ".join(conditions.get("nutrition_high", [])) or "없음"}
- **제한 영양소:** {", ".join(conditions.get("nutrition_low", [])) or "없음"}
- **건강상태:** {", ".join(conditions.get("diseases", [])) or "없음"}
"""

    allergies = conditions.get(
        "allergies",
        [],
    )

    if allergies:
        allergy_text = ", ".join(allergies)
        disclaimer = (
            f"※ 제공된 알레르기 표시정보에 "
            f"**{allergy_text}**가 표시되지 않은 후보를 "
            f"기준으로 추천했습니다. "
            f"표시정보만으로 알레르기 안전성을 보증하지 않습니다."
        )
    else:
        disclaimer = (
            "※ 본 결과는 식단 추천 지원용이며 "
            "의학적 진단·치료를 대체하지 않습니다."
        )

    meta_md = f"""
**추천 후보 처리:** RAG 검색 → 알레르기 Hard Filter → LLM Meal-role Gate → 식단 조합 → Multi-Nutrient Ranking → K-Means Cluster Diversity

**생성 식단 조합:** {len(meal_df):,}개  
**최종 추천:** {len(final_df)}개
"""

    return (
        condition_md,
        result_df,
        meta_md,
        disclaimer,
    )


# ============================================================
# 9. 서버 시작 시 모델 / 데이터 로드
# ============================================================

print("[1/5] 급식/영양 데이터 로드")
RECOMMENDABLE_MENU_DF, MENU_DOCUMENTS_DF, NUTRIENT_COLS = (
    load_service_data()
)

print(
    f"      추천 가능 메뉴: {len(RECOMMENDABLE_MENU_DF):,}"
)

RECOMMENDABLE_MENU_DF, MENU_DOCUMENTS_DF = (
    align_to_saved_documents(
        RECOMMENDABLE_MENU_DF,
        MENU_DOCUMENTS_DF,
    )
)

print("[2/5] BGE-M3 모델 로드")
BGE_MODEL = BGEM3FlagModel(
    BGE_MODEL_NAME,
    use_fp16=False,
)

print("[3/5] 메뉴 embedding / FAISS 준비")
MENU_EMBEDDINGS = load_or_build_embeddings(
    BGE_MODEL,
    MENU_DOCUMENTS_DF,
)

SLOT_MENU_DFS, SLOT_INDEXES = (
    build_slot_indexes(
        RECOMMENDABLE_MENU_DF,
        MENU_EMBEDDINGS,
    )
)

print("[4/5] K-Means 영양 프로파일 학습")
CLUSTER_MODEL = MenuNutritionCluster(
    k_min=3,
    k_max=7,
    random_state=42,
)

CLUSTERED_MENU_DF = (
    CLUSTER_MODEL.fit_predict(
        RECOMMENDABLE_MENU_DF
    )
)

CLUSTERED_MENU_DF["menu_fd_Code"] = (
    CLUSTERED_MENU_DF["menu_fd_Code"]
    .astype(str)
)

CLUSTER_MAP = (
    CLUSTERED_MENU_DF
    .set_index("menu_fd_Code")[
        "nutrition_cluster"
    ]
    .to_dict()
)

print(
    f"      선택 K={CLUSTER_MODEL.best_k_}, "
    f"Silhouette={CLUSTER_MODEL.silhouette_score_:.4f}"
)

print("[5/5] 서비스 준비 완료")


# ============================================================
# 10. Gradio UI
# ============================================================

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
#title {
    text-align: center;
    margin-bottom: 6px;
}
#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 24px;
}
"""


with gr.Blocks(
    title="끼니핏",
    css=CSS,
) as demo:

    gr.Markdown(
        "# 밀핏",
        elem_id="title",
    )

    gr.Markdown(
        "### ML · RAG · LLM 기반 맞춤형 급식 식단 추천",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="급식 조건",
                placeholder=(
                    "예) 요양원 급식이고 대두 알레르기가 있어. "
                    "단백질이 높고 나트륨이 낮은 메뉴를 추천해줘. "
                    "고혈압이 있는 어르신이야."
                ),
                lines=5,
            )

            recommend_btn = gr.Button(
                "맞춤 식단 추천",
                variant="primary",
            )

            gr.Examples(
                examples=[
                    [
                        "요양원 급식이고 대두 알레르기가 있어. "
                        "단백질이 높고 나트륨이 낮은 메뉴를 추천해줘. "
                        "고혈압이 있는 어르신이야."
                    ],
                    [
                        "학생 급식이야. 단백질과 칼슘이 풍부한 "
                        "균형 잡힌 식단을 추천해줘."
                    ],
                    [
                        "성인 급식이고 나트륨과 포화지방은 낮게, "
                        "식이섬유는 높게 추천해줘."
                    ],
                ],
                inputs=user_input,
            )

        with gr.Column(scale=1):
            condition_output = gr.Markdown(
                "### 입력 조건 분석\n추천 조건을 입력해 주세요."
            )

    gr.Markdown("## 추천 식단 TOP 3")

    result_output = gr.Dataframe(
        interactive=False,
        wrap=True,
    )

    process_output = gr.Markdown()
    disclaimer_output = gr.Markdown()

    recommend_btn.click(
        fn=recommend_meals,
        inputs=user_input,
        outputs=[
            condition_output,
            result_output,
            process_output,
            disclaimer_output,
        ],
    )


if __name__ == "__main__":
    server_port = int(
        os.getenv(
            "GRADIO_SERVER_PORT",
            "7861",
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=True,
        show_error=True,
    )
