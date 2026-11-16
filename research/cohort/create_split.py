"""
09_freeze_split.py

Haowei Project - Fixed 7:1:2 split and raw-data manifest freeze

功能：
1. 读取 eligible_cohort_v0.4.csv；
2. 验证合格样本数量、ID、路径和SHA-256；
3. 验证合格队列不存在精确重复图像；
4. 使用确定性SHA-256排序完成7:1:2划分；
5. 精确生成：
   - train: 80,740
   - val: 11,534
   - test: 23,069
6. 验证split之间不存在sample、目录、ID或文件哈希泄漏；
7. 冻结原始图像manifest；
8. 生成split清单、分布报告、dataset card和文件哈希；
9. 不读取或修改原始图像；
10. 不进行预处理。

说明：
- 当前split单位是数字文件夹派生的匿名分组ID；
- 全量精确重复文件已经检查并排除；
- 缺少额外用户映射，因此非完全相同的重复拍摄无法被绝对验证。
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import csv
import hashlib
import json
import shutil


# ============================================================
# 1. 路径与版本
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data_audit"
    / "eligible_cohort_v0.4"
)

INPUT_COHORT = (
    INPUT_DIR
    / "eligible_cohort_v0.4.csv"
)

INPUT_EXCLUSION_LOG = (
    INPUT_DIR
    / "exclusion_log_v0.4.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_protocol"
    / "dataset_v1.0"
)

FROZEN_MANIFEST = (
    OUTPUT_DIR
    / "manifest_frozen_v1.0.csv"
)

SPLIT_GROUPS = (
    OUTPUT_DIR
    / "split_groups_v1.0.csv"
)

EXCLUSION_LOG = (
    OUTPUT_DIR
    / "exclusion_log_v1.0.csv"
)

SPLIT_DISTRIBUTION = (
    OUTPUT_DIR
    / "split_distribution_v1.0.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "split_summary_v1.0.json"
)

DATASET_CARD = (
    OUTPUT_DIR
    / "dataset_card_v1.0.md"
)

ARTIFACT_HASHES = (
    OUTPUT_DIR
    / "artifact_hashes_v1.0.json"
)

DATASET_VERSION = "dataset_v1.0"
SPLIT_VERSION = "split_v1.0"

# 任何时候都不要修改这个salt，否则划分会变化
SPLIT_SALT = "haowei_project_split_v1.0_7_1_2"

EXPECTED_ELIGIBLE = 115343

TRAIN_COUNT = 80740
VAL_COUNT = 11534
TEST_COUNT = 23069

EXPECTED_SPLIT_COUNTS = {
    "train": TRAIN_COUNT,
    "val": VAL_COUNT,
    "test": TEST_COUNT,
}

MODALITIES = [
    "rgb",
    "uv",
    "plv",
    "plh",
]


# ============================================================
# 2. 工具函数
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def calculate_sha256(
    path,
    chunk_size=1024 * 1024,
):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def deterministic_split_key(row):
    """
    使用固定salt和稳定ID生成确定性伪随机排序键。
    重复运行可得到完全相同的划分。
    """

    source = (
        f"{SPLIT_SALT}|"
        f"{row['subject_id']}|"
        f"{row['sample_id']}"
    )

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def validate_sha256_text(value):
    if len(value) != 64:
        return False

    try:
        int(value, 16)
        return True
    except Exception:
        return False


def derive_year(capture_date):
    if len(capture_date) >= 4:
        return capture_date[:4]

    return "unknown"


# ============================================================
# 3. 输入验证
# ============================================================

def validate_input_rows(rows):
    if len(rows) != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            "合格队列数量不符合预期："
            f"expected={EXPECTED_ELIGIBLE}; "
            f"observed={len(rows)}"
        )

    sample_counter = Counter()
    subject_counter = Counter()
    directory_counter = Counter()
    capture_counter = Counter()
    path_counter = Counter()
    hash_usage = defaultdict(list)

    invalid_rows = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        sample_id = row["sample_id"].strip()
        subject_id = row["subject_id"].strip()
        sample_directory = (
            row["sample_directory"].strip()
        )
        capture_id = row["capture_id"].strip()

        sample_counter[sample_id] += 1
        subject_counter[subject_id] += 1
        directory_counter[sample_directory] += 1
        capture_counter[capture_id] += 1

        if row["eligible_primary"].strip() != "1":
            invalid_rows.append(
                (
                    row_number,
                    sample_id,
                    "eligible_primary_not_1",
                )
            )

        if not sample_id:
            invalid_rows.append(
                (
                    row_number,
                    sample_id,
                    "empty_sample_id",
                )
            )

        if not subject_id:
            invalid_rows.append(
                (
                    row_number,
                    sample_id,
                    "empty_subject_id",
                )
            )

        if not sample_directory:
            invalid_rows.append(
                (
                    row_number,
                    sample_id,
                    "empty_sample_directory",
                )
            )

        for modality in MODALITIES:
            path_field = f"{modality}_path"
            hash_field = f"{modality}_sha256"

            relative_path = (
                row[path_field].strip()
            )

            hash_value = (
                row[hash_field]
                .strip()
                .casefold()
            )

            if not relative_path:
                invalid_rows.append(
                    (
                        row_number,
                        sample_id,
                        f"empty_{path_field}",
                    )
                )
            else:
                path_counter[relative_path] += 1

            if not validate_sha256_text(
                hash_value
            ):
                invalid_rows.append(
                    (
                        row_number,
                        sample_id,
                        f"invalid_{hash_field}",
                    )
                )
            else:
                hash_usage[hash_value].append({
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "modality": modality,
                    "path": relative_path,
                })

    if invalid_rows:
        first_errors = invalid_rows[:20]

        raise RuntimeError(
            "发现非法合格记录。前20项：\n"
            + "\n".join(
                str(item)
                for item in first_errors
            )
        )

    duplicate_samples = {
        value: count
        for value, count in sample_counter.items()
        if count > 1
    }

    duplicate_subjects = {
        value: count
        for value, count in subject_counter.items()
        if count > 1
    }

    duplicate_directories = {
        value: count
        for value, count in directory_counter.items()
        if count > 1
    }

    duplicate_capture_ids = {
        value: count
        for value, count in capture_counter.items()
        if count > 1
    }

    duplicate_paths = {
        value: count
        for value, count in path_counter.items()
        if count > 1
    }

    exact_duplicate_hashes = {
        hash_value: usages
        for hash_value, usages
        in hash_usage.items()
        if len(usages) > 1
    }

    if duplicate_samples:
        raise RuntimeError(
            "eligible cohort中存在重复sample_id"
        )

    if duplicate_subjects:
        raise RuntimeError(
            "eligible cohort中存在重复subject_id"
        )

    if duplicate_directories:
        raise RuntimeError(
            "eligible cohort中存在重复目录"
        )

    if duplicate_capture_ids:
        raise RuntimeError(
            "eligible cohort中存在重复数字ID"
        )

    if duplicate_paths:
        raise RuntimeError(
            "eligible cohort中存在重复路径"
        )

    if exact_duplicate_hashes:
        examples = list(
            exact_duplicate_hashes.items()
        )[:10]

        raise RuntimeError(
            "合格队列中仍存在精确重复文件：\n"
            + "\n".join(
                f"{hash_value}: {usages}"
                for hash_value, usages
                in examples
            )
        )

    return {
        "sample_count": len(sample_counter),
        "subject_count": len(subject_counter),
        "directory_count": len(
            directory_counter
        ),
        "capture_id_count": len(
            capture_counter
        ),
        "image_path_count": len(
            path_counter
        ),
        "unique_image_hash_count": len(
            hash_usage
        ),
        "exact_duplicate_hash_count": len(
            exact_duplicate_hashes
        ),
    }


# ============================================================
# 4. 固定划分
# ============================================================

def assign_splits(rows):
    ranked_rows = []

    for row in rows:
        new_row = dict(row)
        new_row["split_key_sha256"] = (
            deterministic_split_key(row)
        )

        ranked_rows.append(new_row)

    ranked_rows.sort(
        key=lambda row: (
            row["split_key_sha256"],
            row["sample_id"],
        )
    )

    for index, row in enumerate(
        ranked_rows
    ):
        if index < TRAIN_COUNT:
            split = "train"

        elif index < TRAIN_COUNT + VAL_COUNT:
            split = "val"

        else:
            split = "test"

        row["split"] = split
        row["split_group_id"] = (
            row["subject_id"]
        )
        row["split_unit"] = (
            "numeric_folder_record"
        )
        row["split_version"] = (
            SPLIT_VERSION
        )
        row["dataset_version"] = (
            DATASET_VERSION
        )

    # 最终manifest按照sample_id排序，
    # 而不是按照随机split key排序
    ranked_rows.sort(
        key=lambda row: row["sample_id"]
    )

    return ranked_rows


# ============================================================
# 5. 泄漏检查
# ============================================================

def validate_splits(rows):
    split_samples = defaultdict(set)
    split_subjects = defaultdict(set)
    split_directories = defaultdict(set)
    split_capture_ids = defaultdict(set)
    split_hashes = defaultdict(set)

    for row in rows:
        split = row["split"]

        split_samples[split].add(
            row["sample_id"]
        )

        split_subjects[split].add(
            row["subject_id"]
        )

        split_directories[split].add(
            row["sample_directory"]
        )

        split_capture_ids[split].add(
            row["capture_id"]
        )

        for modality in MODALITIES:
            split_hashes[split].add(
                row[f"{modality}_sha256"]
            )

    observed_counts = {
        split: len(
            split_samples[split]
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    }

    if observed_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "划分数量不符合预期："
            f"expected="
            f"{EXPECTED_SPLIT_COUNTS}; "
            f"observed={observed_counts}"
        )

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    leakage_results = {}

    for split_a, split_b in pairs:
        pair_name = (
            f"{split_a}_vs_{split_b}"
        )

        sample_overlap = (
            split_samples[split_a]
            & split_samples[split_b]
        )

        subject_overlap = (
            split_subjects[split_a]
            & split_subjects[split_b]
        )

        directory_overlap = (
            split_directories[split_a]
            & split_directories[split_b]
        )

        capture_overlap = (
            split_capture_ids[split_a]
            & split_capture_ids[split_b]
        )

        hash_overlap = (
            split_hashes[split_a]
            & split_hashes[split_b]
        )

        leakage_results[pair_name] = {
            "sample_overlap": len(
                sample_overlap
            ),
            "subject_overlap": len(
                subject_overlap
            ),
            "directory_overlap": len(
                directory_overlap
            ),
            "capture_id_overlap": len(
                capture_overlap
            ),
            "exact_image_hash_overlap": len(
                hash_overlap
            ),
        }

        if any([
            sample_overlap,
            subject_overlap,
            directory_overlap,
            capture_overlap,
            hash_overlap,
        ]):
            raise RuntimeError(
                "发现split泄漏："
                f"{pair_name}; "
                f"{leakage_results[pair_name]}"
            )

    return {
        "split_counts": observed_counts,
        "leakage_checks": leakage_results,
    }


# ============================================================
# 6. 分布统计
# ============================================================

def build_distribution_rows(rows):
    counters = defaultdict(Counter)

    for row in rows:
        split = row["split"]
        year = derive_year(
            row["capture_date"]
        )
        resolution = (
            f"{row['source_width']}x"
            f"{row['source_height']}"
        )

        counters[
            ("split_total", split)
        ]["all"] += 1

        counters[
            ("capture_year", split)
        ][year] += 1

        counters[
            ("source_resolution", split)
        ][resolution] += 1

    output_rows = []

    for (
        variable,
        split,
    ), counter in sorted(
        counters.items()
    ):
        for value, count in sorted(
            counter.items()
        ):
            output_rows.append({
                "variable": variable,
                "value": value,
                "split": split,
                "sample_count": count,
            })

    return output_rows


# ============================================================
# 7. Dataset card
# ============================================================

def build_dataset_card(
    input_hash,
    manifest_hash,
    split_hash,
    exclusion_hash,
    validation,
    split_validation,
):
    text = f"""# Haowei Project Dataset Card v1.0

## Scope

The project uses four paired image modalities:

- RGB
- UV
- PLV
- PLH

## Source cohort

- Complete four-modality candidate samples: 115,382
- Excluded black or constant target samples: 38
- Excluded later exact four-modality duplicate: 1
- Eligible samples: 115,343

## Image integrity

- All 461,528 candidate images passed full JPEG decoding.
- No full-decode failures were observed.
- Exact SHA-256 hashes were computed for every candidate image.
- No exact duplicate image remains in the eligible cohort.

## Split

The fixed split ratio is 7:1:2:

- Train: 80,740
- Validation: 11,534
- Test: 23,069

Split version: `{SPLIT_VERSION}`

Deterministic split salt:

`{SPLIT_SALT}`

The split was generated by deterministic SHA-256 ordering. It must not
be regenerated with another salt.

## Split unit and identity limitation

The second-level numeric directory is used as the fixed grouping unit.
A pseudonymous folder-derived identifier is stored in `subject_id`.

The available export contains one numeric ID per sample. The data source
indicates that the folder represents a user record, and the dataset has
previously been used in related projects.

However, no independent user-to-capture mapping was available. Therefore,
non-identical repeat captures of the same person cannot be mathematically
ruled out across folders.

All exact duplicate files and the known exact four-modality duplicate
record were removed before splitting.

## Leakage checks

All checks passed:

- No sample ID overlap between splits
- No pseudonymous folder-ID overlap between splits
- No source-directory overlap between splits
- No numeric folder-ID overlap between splits
- No exact image SHA-256 overlap between splits
- All four modalities of a sample share one split

## Versioning

- Dataset version: `{DATASET_VERSION}`
- Split version: `{SPLIT_VERSION}`
- Input eligible-cohort SHA-256:
  `{input_hash}`
- Frozen manifest SHA-256:
  `{manifest_hash}`
- Split-group file SHA-256:
  `{split_hash}`
- Exclusion log SHA-256:
  `{exclusion_hash}`

## Raw-data policy

No raw image was modified, moved, renamed, or deleted.

## Preprocessing status

Image preprocessing is not yet frozen in this dataset card. A separate
versioned preprocessing protocol will define face ROI, resize,
normalization, and augmentation.
"""

    DATASET_CARD.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():
    if not INPUT_COHORT.exists():
        raise FileNotFoundError(
            f"合格队列不存在：{INPUT_COHORT}"
        )

    if not INPUT_EXCLUSION_LOG.exists():
        raise FileNotFoundError(
            f"排除日志不存在："
            f"{INPUT_EXCLUSION_LOG}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Haowei Project：固定7:1:2划分并冻结Manifest")
    print("=" * 72)
    print(f"Input cohort: {INPUT_COHORT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Split salt: {SPLIT_SALT}")
    print()

    input_rows = read_csv(
        INPUT_COHORT
    )

    print("Step 1/6：验证合格队列")

    validation = validate_input_rows(
        input_rows
    )

    for key, value in validation.items():
        print(f"{key}: {value}")

    print()
    print("Step 2/6：执行确定性7:1:2划分")

    split_rows = assign_splits(
        input_rows
    )

    print("Step 3/6：检查所有类型的split泄漏")

    split_validation = validate_splits(
        split_rows
    )

    print(
        "split_counts:",
        split_validation["split_counts"],
    )

    print(
        "leakage_checks:",
        split_validation[
            "leakage_checks"
        ],
    )

    print()
    print("Step 4/6：写入冻结Manifest和分组文件")

    manifest_fieldnames = list(
        input_rows[0].keys()
    )

    for added_field in [
        "split_key_sha256",
        "split_group_id",
        "split_unit",
        "split_version",
    ]:
        if added_field not in manifest_fieldnames:
            manifest_fieldnames.append(
                added_field
            )

    write_csv(
        FROZEN_MANIFEST,
        split_rows,
        manifest_fieldnames,
    )

    split_group_rows = [
        {
            "split_group_id": (
                row["split_group_id"]
            ),
            "subject_id": row["subject_id"],
            "capture_id": row["capture_id"],
            "sample_id": row["sample_id"],
            "sample_directory": (
                row["sample_directory"]
            ),
            "capture_date": (
                row["capture_date"]
            ),
            "split": row["split"],
            "split_key_sha256": (
                row["split_key_sha256"]
            ),
            "split_unit": row["split_unit"],
            "split_version": (
                row["split_version"]
            ),
        }
        for row in split_rows
    ]

    write_csv(
        SPLIT_GROUPS,
        split_group_rows,
        [
            "split_group_id",
            "subject_id",
            "capture_id",
            "sample_id",
            "sample_directory",
            "capture_date",
            "split",
            "split_key_sha256",
            "split_unit",
            "split_version",
        ],
    )

    shutil.copy2(
        INPUT_EXCLUSION_LOG,
        EXCLUSION_LOG,
    )

    distribution_rows = (
        build_distribution_rows(
            split_rows
        )
    )

    write_csv(
        SPLIT_DISTRIBUTION,
        distribution_rows,
        [
            "variable",
            "value",
            "split",
            "sample_count",
        ],
    )

    print("Step 5/6：生成哈希和Dataset Card")

    input_hash = calculate_sha256(
        INPUT_COHORT
    )

    manifest_hash = calculate_sha256(
        FROZEN_MANIFEST
    )

    split_hash = calculate_sha256(
        SPLIT_GROUPS
    )

    exclusion_hash = calculate_sha256(
        EXCLUSION_LOG
    )

    build_dataset_card(
        input_hash=input_hash,
        manifest_hash=manifest_hash,
        split_hash=split_hash,
        exclusion_hash=exclusion_hash,
        validation=validation,
        split_validation=(
            split_validation
        ),
    )

    dataset_card_hash = (
        calculate_sha256(
            DATASET_CARD
        )
    )

    artifact_hashes = {
        "eligible_cohort_v0.4.csv": (
            input_hash
        ),
        "manifest_frozen_v1.0.csv": (
            manifest_hash
        ),
        "split_groups_v1.0.csv": (
            split_hash
        ),
        "exclusion_log_v1.0.csv": (
            exclusion_hash
        ),
        "dataset_card_v1.0.md": (
            dataset_card_hash
        ),
    }

    write_json(
        ARTIFACT_HASHES,
        artifact_hashes,
    )

    print("Step 6/6：生成最终汇总")

    split_resolution = defaultdict(
        Counter
    )

    split_year = defaultdict(
        Counter
    )

    for row in split_rows:
        split = row["split"]

        resolution = (
            f"{row['source_width']}x"
            f"{row['source_height']}"
        )

        year = derive_year(
            row["capture_date"]
        )

        split_resolution[split][
            resolution
        ] += 1

        split_year[split][year] += 1

    summary = {
        "dataset_version": (
            DATASET_VERSION
        ),
        "split_version": (
            SPLIT_VERSION
        ),
        "created_at_utc": utc_now(),
        "input_cohort": str(
            INPUT_COHORT
        ),
        "input_cohort_sha256": (
            input_hash
        ),
        "eligible_sample_count": len(
            split_rows
        ),
        "split_ratio": {
            "train": 0.7,
            "val": 0.1,
            "test": 0.2,
        },
        "split_counts": (
            split_validation[
                "split_counts"
            ]
        ),
        "split_salt": SPLIT_SALT,
        "split_method": (
            "deterministic_sha256_ordering"
        ),
        "split_unit": (
            "numeric_folder_record"
        ),
        "unique_sample_count": (
            validation["sample_count"]
        ),
        "unique_group_count": (
            validation["subject_count"]
        ),
        "unique_image_path_count": (
            validation[
                "image_path_count"
            ]
        ),
        "unique_image_hash_count": (
            validation[
                "unique_image_hash_count"
            ]
        ),
        "remaining_exact_duplicate_hashes": (
            validation[
                "exact_duplicate_hash_count"
            ]
        ),
        "leakage_checks": (
            split_validation[
                "leakage_checks"
            ]
        ),
        "resolution_distribution_by_split": {
            split: dict(
                sorted(counter.items())
            )
            for split, counter
            in sorted(
                split_resolution.items()
            )
        },
        "year_distribution_by_split": {
            split: dict(
                sorted(counter.items())
            )
            for split, counter
            in sorted(
                split_year.items()
            )
        },
        "manifest_frozen": True,
        "raw_data_modified": False,
        "preprocessing_frozen": False,
        "training_started": False,
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print("=" * 72)
    print("固定划分和Manifest冻结完成")
    print("=" * 72)

    print(
        "split_counts:",
        summary["split_counts"],
    )

    print(
        "remaining_exact_duplicate_hashes:",
        summary[
            "remaining_exact_duplicate_hashes"
        ],
    )

    print(
        "manifest_frozen:",
        summary["manifest_frozen"],
    )

    print(
        "raw_data_modified:",
        summary["raw_data_modified"],
    )

    print()
    print("主要输出：")
    print(FROZEN_MANIFEST)
    print(SPLIT_GROUPS)
    print(EXCLUSION_LOG)
    print(SPLIT_DISTRIBUTION)
    print(SUMMARY_JSON)
    print(DATASET_CARD)
    print(ARTIFACT_HASHES)


if __name__ == "__main__":
    main()