from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_dictator_corrected_baselines import load_explicit_behavior, select_people, summarize_contrasts


def behavior() -> pd.DataFrame:
    return pd.DataFrame([dict(Dataset=dataset,Nation=nation,SubNo=person,GroupNo=item,allocation_percent=(item%21)*5,Token=83) for dataset in range(1,5) for nation in range(2) for person in range(1,4) for item in range(1,26)])


def test_response_column_is_explicit_and_an_off_grid_rating_is_rejected(tmp_path) -> None:
    path=tmp_path/"behavior.csv"
    behavior().to_csv(path,index=False)
    frame=load_explicit_behavior(path,"allocation_percent","percent")
    assert frame.TokenChoice.iloc[0]==1
    with pytest.raises(ValueError,match="refusing to bin"):
        load_explicit_behavior(path,"Token","percent")


def test_missing_or_duplicate_allocation_rows_are_rejected(tmp_path) -> None:
    path=tmp_path/"behavior.csv"
    behavior().iloc[1:].to_csv(path,index=False)
    with pytest.raises(ValueError,match="25 items"):
        load_explicit_behavior(path,"allocation_percent","percent")
    pd.concat([behavior(),behavior().iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="Duplicate"):
        load_explicit_behavior(path,"allocation_percent","percent")


def test_manifest_join_selects_only_complete_requested_people(tmp_path) -> None:
    path=tmp_path/"behavior.csv"
    behavior().to_csv(path,index=False)
    frame=load_explicit_behavior(path,"allocation_percent","percent")
    manifest=pd.DataFrame([dict(Dataset=1,Nation=0,SubNo=2,split="test"),dict(Dataset=1,Nation=0,SubNo=1,split="train")])
    selected=select_people(frame,manifest,"test")
    assert len(selected)==25 and selected.SubNo.eq(2).all()


def test_frozen_control_contrasts_do_not_introduce_existing_adapters() -> None:
    rows=[]
    for seed in range(3):
        for nation,country in [(0,"United States"),(1,"China")]:
            for person in range(1,5):
                for model,offset in [("llama",0),("centaur",.2)]:
                    for variant,label_offset in [("original",0),("us",.03),("china",.05)]:
                        rows.append(dict(seed=seed,Dataset=3,Nation=nation,SubNo=person,Country=country,task="Taking",condition=f"{model}_{variant}",raw_response_ll=-3+offset+label_offset,conditional_ll=-3))
    _,summary=summarize_contrasts(pd.DataFrame(rows),100,2)
    raw=summary.loc[summary.metric.eq("raw_response_ll")]
    np.testing.assert_allclose(raw.loc[raw.contrast.eq("centaur_minus_llama_original"),"mean"],.2)
    np.testing.assert_allclose(summary.loc[summary.metric.eq("conditional_ll"),"mean"],0)
    assert not summary.contrast.str.contains("adapter").any()
