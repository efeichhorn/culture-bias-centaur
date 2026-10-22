import pytest

from scripts.prepare_culture_extension_runs import aggregate_targets, make_splits


@pytest.mark.parametrize("dataset,n_us,n_cn,train_n,val_n,test_n", [
    ("cao",295,178,124,26,28), ("anllo",50,49,34,7,8)
])
def test_matched_population_budgets_and_disjoint_people(dataset,n_us,n_cn,train_n,val_n,test_n):
    people=[{"participant_id":f"{c}-{i}","country":c} for c,n in [("us",n_us),("china",n_cn)] for i in range(n)]
    split,conditions=make_splits(people,dataset,0)
    assignment={r["participant_id"]:r["split"] for r in split}
    assert len(assignment)==len(people)
    for c in ["us","china"]:
        for part,n in [("train",train_n),("validation",val_n),("test",test_n)]:
            assert sum(r["country"]==c and r["split"]==part for r in split)==n
        assert len(conditions[c])==train_n
        assert all(assignment[r["participant_id"]]=="train" for r in conditions[c])
        pooled={r["participant_id"] for r in conditions["pooled"] if r["country"]==c}
        assert len(pooled)==train_n//2
        assert pooled <= {r["participant_id"] for r in conditions[c]}
    assert (split,conditions)==make_splits(people,dataset,0)


def test_targets_exclude_both_unselected_people_and_heldout_items():
    selected=[{"participant_id":"train","country":"us"}]
    rows=[{"participant_id":p,"country":"us","item_id":item,"response_positive":value}
          for p,item,value in [("train","seen",1),("test","seen",0),("train","unseen",0)]]
    assert aggregate_targets(rows,selected,["seen"],1)==[
        {"item_id":"seen","p_positive":1.0,"n_responses":1,"n_participants":1}]


def test_incomplete_lottery_repetitions_are_rejected():
    selected=[{"participant_id":"train","country":"china"}]
    rows=[{"participant_id":"train","country":"china","item_id":"8","response_positive":1} for _ in range(3)]
    with pytest.raises(ValueError,match="repetition"):
        aggregate_targets(rows,selected,["8"],4)
