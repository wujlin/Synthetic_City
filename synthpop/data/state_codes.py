from __future__ import annotations

"""US state and District of Columbia identifier helpers."""

STATEFP_TO_POSTAL: dict[str, str] = {
    "01": "al",
    "02": "ak",
    "04": "az",
    "05": "ar",
    "06": "ca",
    "08": "co",
    "09": "ct",
    "10": "de",
    "11": "dc",
    "12": "fl",
    "13": "ga",
    "15": "hi",
    "16": "id",
    "17": "il",
    "18": "in",
    "19": "ia",
    "20": "ks",
    "21": "ky",
    "22": "la",
    "23": "me",
    "24": "md",
    "25": "ma",
    "26": "mi",
    "27": "mn",
    "28": "ms",
    "29": "mo",
    "30": "mt",
    "31": "ne",
    "32": "nv",
    "33": "nh",
    "34": "nj",
    "35": "nm",
    "36": "ny",
    "37": "nc",
    "38": "nd",
    "39": "oh",
    "40": "ok",
    "41": "or",
    "42": "pa",
    "44": "ri",
    "45": "sc",
    "46": "sd",
    "47": "tn",
    "48": "tx",
    "49": "ut",
    "50": "vt",
    "51": "va",
    "53": "wa",
    "54": "wv",
    "55": "wi",
    "56": "wy",
}

POSTAL_TO_STATEFP: dict[str, str] = {v: k for k, v in STATEFP_TO_POSTAL.items()}


def statefp_to_postal(statefp: str) -> str:
    key = str(statefp).zfill(2)
    try:
        return STATEFP_TO_POSTAL[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported statefp: {statefp}") from exc
