# Fortran 固定形式版アーカイブ

## 概要

このディレクトリには YU 動力学硬化 UMAT の固定形式（`.for`）版が保管されている。
これらは `fortran/yu_kinematic_3d.f90`（自由形式）から派生したもので、
コンパイラの型定義の懸念から調査目的で作成された。

**現在の正式採用ファイル**: `fortran/yu_kinematic_3d.f90`（自由形式、gfortran/ABAQUS 共用）

---

## ファイル

### `yu_kinematic_3d_fixed.for`

`yu_kinematic_3d.f90` を固定形式 72 文字制限に変換したもの。
Python 変換スクリプト（`&` 継続行の結合 → 66 文字で再分割）で生成。
`-m yu_kinematic_3d_fixed` として f2py でビルド可能。

### `yu_kinematic_3d_abaqus.for`

ABAQUS 公式インターフェースに完全準拠した固定形式版。以下の変更を適用：

1. **UMAT エントリポイント**: 公式インターフェース採用
   - `INCLUDE 'ABA_PARAM.INC'`（`implicit real*8(a-h,o-z)` + `nprecd=2`）
   - `JSTEP(4)`（スカラー `KSTEP` の代わり。ABAQUS マニュアル準拠）
   - `DIMENSION` のみ宣言、`implicit none` なし
   - 全処理を `UMAT_IMPL` サブルーチンに委譲

2. **UMAT_IMPL**: `implicit none` + `real*8` で型安全な実装
   - `KSTEP = JSTEP(1)` として受け取り

3. **計算サブルーチン群**: `double precision` → `real*8` に統一

---

## 調査結果と結論

### 数値比較

自由形式（`.f90`）と固定形式（`.for`）を同一入力に対して比較した結果、
**全ての比較で差ゼロ（機械精度で完全一致）** を確認した。

```
Python vs free-form (.f90):   max diff = 7.11e-15
Python vs abaqus-form (.for): max diff = 7.11e-15
free-form vs abaqus-form:     max diff = 0.00e+00
```

### ABA_PARAM.INC の型定義影響

当初、`implicit real*8(a-h,o-z)` が `double precision` 宣言と競合し、型定義の
問題が起きる可能性を懸念した。しかし調査の結果：

- `UMAT` は `ABA_PARAM.INC` を `include` し暗黙型宣言に従う
- `UMAT_IMPL` は `implicit none` + `real*8` 宣言で型安全
- 数値結果は自由形式と完全一致

### 自由形式採用の理由

1. **可読性**: 72 文字制限がなく、長い変数名・式が書ける
2. **保守性**: 唯一のソースファイル。固定形式への変換は自動化できるが誤変換リスクがある
3. **gfortran/ABAQUS 共用**: gfortran は自由形式 `.f90` を問題なく受け付ける。
   ABAQUS も `-free` コンパイルオプションまたは拡張子 `.f90` で自由形式を処理可能
4. **テスト基盤**: f2py、pytest ともに `.f90` で直接動作確認済み

### ABAQUS への組み込み方法（`.f90` のまま使う場合）

ABAQUS では通常 `.for` が固定形式として扱われるが、以下のいずれかで `.f90` を使用可能：

- **job.env** に `compile_fortran += ['-free']` を追加
- ファイル名を `.f` にして環境設定で自由形式を指定
- または本アーカイブの `yu_kinematic_3d_abaqus.for` を使用（内容は同一）

---

## 再生成方法（必要な場合）

```python
# 自由形式 → 固定形式変換スクリプト（過去のセッションで使用）
# free_to_fixed('fortran/yu_kinematic_3d.f90', 'archives/fortran_fixed_form/yu_kinematic_3d_fixed.for')
# その後 abaqus.for は fixed.for をベースに UMAT/UMAT_IMPL 構造に変換
```
