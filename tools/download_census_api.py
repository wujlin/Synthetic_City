#!/usr/bin/env python3
"""
下载 2020 DHC (Decennial Census) 和 ACS 5-Year Summary Tables
需要 Census API key：https://api.census.gov/data/key_signup.html

用法:
    python tools/download_census_api.py --api-key YOUR_KEY --output-dir dataset/census

输出:
    dataset/census/dhc_2020_bg_michigan.csv.gz          # DHC P12/P5 at Block Group
    dataset/census/acs5_2022_B01001_tract_michigan.csv.gz
    dataset/census/acs5_2022_B15003_tract_michigan.csv.gz
    dataset/census/acs5_2022_B19001_tract_michigan.csv.gz
    dataset/census/acs5_2022_B20001_tract_michigan.csv.gz
    dataset/census/acs5_2022_B23025_tract_michigan.csv.gz
"""

import argparse
import os
import time
import requests
import pandas as pd
from pathlib import Path


# ── Michigan FIPS ──
STATE_FIPS = "26"


# ═══════════════════════════════════════════════════════════════════
# Part 1: 2020 DHC (Decennial Census) at Block Group
# ═══════════════════════════════════════════════════════════════════

# DHC 表名和变量范围
# P12: Sex by Age (total) — P12_001N to P12_049N (49 vars)
# P5:  Hispanic or Latino Origin by Race — P5_001N to P5_017N (17 vars)
# PCT12: Sex by Single Year of Age (更细粒度, 可选)

DHC_TABLES = {
    "P12": {
        "description": "Sex by Age (total population)",
        "prefix": "P12",
        "count": 49,  # P12_001N to P12_049N
    },
    "P5": {
        "description": "Hispanic or Latino Origin by Race",
        "prefix": "P5",
        "count": 17,  # P5_001N to P5_017N
    },
}


def build_dhc_var_names(prefix: str, count: int) -> list[str]:
    """生成变量名列表，如 P12_001N, P12_002N, ..., P12_049N"""
    return [f"{prefix}_{str(i).zfill(3)}N" for i in range(1, count + 1)]


def fetch_dhc_block_group(api_key: str) -> pd.DataFrame:
    """
    从 Census API 获取全 Michigan 所有 Block Group 的 DHC 数据。
    
    地理层级: block group:* in state:26
    分批请求（每批 ≤ 50 变量，API 限制）
    """
    base_url = "https://api.census.gov/data/2020/dec/dhc"
    
    # 汇总所有需要的变量
    all_vars = []
    for table_name, table_info in DHC_TABLES.items():
        vars_list = build_dhc_var_names(table_info["prefix"], table_info["count"])
        all_vars.extend(vars_list)
        print(f"  表 {table_name} ({table_info['description']}): {len(vars_list)} 个变量")
    
    print(f"  总计需要获取 {len(all_vars)} 个变量")
    
    # 地理标识变量（始终请求）
    geo_vars = ["NAME"]
    
    # 分批请求，每批最多 48 个数据变量（留 2 个位置给 geo 标识）
    batch_size = 48
    batches = [all_vars[i:i+batch_size] for i in range(0, len(all_vars), batch_size)]
    
    print(f"  分 {len(batches)} 批请求...")
    
    dfs = []
    for i, batch_vars in enumerate(batches):
        var_str = ",".join(geo_vars + batch_vars)
        url = (
            f"{base_url}?get={var_str}"
            f"&for=block%20group:*"
            f"&in=state:{STATE_FIPS}%20county:*"
            f"&key={api_key}"
        )
        
        print(f"  批次 {i+1}/{len(batches)}: {len(batch_vars)} 个变量...", end=" ")
        
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"OK, {len(df)} 行")
        
        dfs.append(df)
        
        # 避免频率限制
        if i < len(batches) - 1:
            time.sleep(1)
    
    # 合并所有批次（按地理标识列合并）
    geo_cols = ["NAME", "state", "county", "tract", "block group"]
    result = dfs[0]
    for df in dfs[1:]:
        # 只保留数据列（不重复 geo 列）
        data_cols = [c for c in df.columns if c not in geo_cols]
        result = result.merge(
            df[geo_cols + data_cols],
            on=geo_cols,
            how="outer",
        )
    
    # 构造 GEOID (state + county + tract + block group)
    result["GEOID"] = (
        result["state"] + result["county"] + result["tract"] + result["block group"]
    )
    
    # 将数据列转为数值型
    data_cols = [c for c in result.columns if c not in geo_cols + ["GEOID"]]
    for col in data_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Part 2: ACS 5-Year Summary Tables at Tract
# ═══════════════════════════════════════════════════════════════════

# 使用 2022 ACS 5-Year（与 PUMS 一致）
ACS_YEAR = "2022"

ACS_TABLES = {
    "B01001": {
        "description": "Sex by Age",
        "count": 49,  # B01001_001E to B01001_049E
    },
    "B15003": {
        "description": "Educational Attainment (25+)",
        "count": 25,  # B15003_001E to B15003_025E
    },
    "B19001": {
        "description": "Household Income (16 bins)",
        "count": 17,  # B19001_001E to B19001_017E
    },
    "B20001": {
        "description": "Earnings (person-level, for validation against PINCP)",
        "count": 43,  # B20001_001E to B20001_043E (Sex by Earnings)
    },
    "B23025": {
        "description": "Employment Status (16+)",
        "count": 7,   # B23025_001E to B23025_007E
    },
}


def build_acs_var_names(prefix: str, count: int) -> list[str]:
    """生成 ACS 变量名列表，如 B01001_001E, B01001_002E, ..."""
    return [f"{prefix}_{str(i).zfill(3)}E" for i in range(1, count + 1)]


def fetch_acs_tract_table(table_name: str, table_info: dict, api_key: str) -> pd.DataFrame:
    """
    获取单张 ACS Summary Table 的全 Michigan tract 级别数据。
    """
    base_url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
    
    vars_list = build_acs_var_names(table_name, table_info["count"])
    geo_vars = ["NAME"]
    
    # ACS 表变量一般不超过 50，单次请求即可
    var_str = ",".join(geo_vars + vars_list)
    
    url = (
        f"{base_url}?get={var_str}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}"
        f"&key={api_key}"
    )
    
    print(f"  请求 {table_name} ({table_info['description']})...", end=" ")
    
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # 构造 GEOID
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    
    # 转数值
    for col in vars_list:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"OK, {len(df)} tracts")
    return df


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="下载 Census DHC 和 ACS Summary Tables for Michigan"
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Census API key (从 https://api.census.gov/data/key_signup.html 获取)"
    )
    parser.add_argument(
        "--output-dir", default="dataset/census",
        help="输出目录 (默认: dataset/census)"
    )
    parser.add_argument(
        "--skip-dhc", action="store_true",
        help="跳过 DHC 下载"
    )
    parser.add_argument(
        "--skip-acs", action="store_true",
        help="跳过 ACS 下载"
    )
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Part 1: DHC ──
    if not args.skip_dhc:
        print("\n" + "=" * 60)
        print("Part 1: 下载 2020 DHC at Block Group (全 Michigan)")
        print("=" * 60)
        
        dhc_path = out_dir / "dhc_2020_bg_michigan.csv.gz"
        if dhc_path.exists():
            print(f"  已存在: {dhc_path}，跳过")
        else:
            df = fetch_dhc_block_group(args.api_key)
            df.to_csv(dhc_path, index=False, compression="gzip")
            print(f"\n  保存到: {dhc_path}")
            print(f"  行数: {len(df)} (Block Groups)")
            print(f"  列数: {len(df.columns)}")
            
            # 简单校验
            total_pop = df["P12_001N"].sum()
            print(f"  Michigan 总人口 (P12_001N sum): {total_pop:,.0f}")
            print(f"  (2020 Census: 约 10,077,331)")
    
    # ── Part 2: ACS Summary Tables ──
    if not args.skip_acs:
        print("\n" + "=" * 60)
        print(f"Part 2: 下载 {ACS_YEAR} ACS 5-Year Summary Tables at Tract (全 Michigan)")
        print("=" * 60)
        
        for table_name, table_info in ACS_TABLES.items():
            acs_path = out_dir / f"acs5_{ACS_YEAR}_{table_name}_tract_michigan.csv.gz"
            if acs_path.exists():
                print(f"  已存在: {acs_path}，跳过")
                continue
            
            df = fetch_acs_tract_table(table_name, table_info, args.api_key)
            df.to_csv(acs_path, index=False, compression="gzip")
            print(f"    保存到: {acs_path} ({len(df)} tracts)")
            
            time.sleep(1)  # 避免频率限制
    
    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("下载汇总")
    print("=" * 60)
    
    for f in sorted(out_dir.glob("*.csv.gz")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
