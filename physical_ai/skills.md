# Available Skills

## PICK

指定されたobjectを把持する。

### Parameters
- `target_id`: 把持するobjectのID

### Preconditions
- target objectが存在する
- target objectをまだ把持していない

### Effects
- target objectを把持している状態になる


## PLACE

把持しているobjectを、reference objectとの指定された位置関係になるように置く。

### Parameters
- `object_id`: 置くobjectのID
- `relation`: 目標とする位置関係
- `reference_id`: 基準となるobjectのID

### Preconditions
- objectを把持している
- reference objectが存在する

### Effects
- objectを把持していない状態になる
- objectが指定された位置関係になる
