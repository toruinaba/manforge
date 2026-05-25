# tests/benchmarks/yu_kinematic/

Numerical-equivalence harness for the Yoshida-Uemori two-surface + stagnation-surface
kinematic hardening model (`YUKinematic3D / YUKinematicPS / YUKinematic1D`).

## テスト対象とパス

| パス | ファイル | 内容 |
|---|---|---|
| Path A | `test_analytical_vs_numerical.py` | 解析的 return mapping (`user_defined_return_mapping`) vs autograd NR の数値等価性検証 |
| Path B | `test_numerical_vs_fortran.py` | `PythonAnalyticalIntegrator` vs `FortranIntegrator` (`yu_kinematic_3d` UMAT) の数値等価性検証 |

`user_defined_return_mapping` / `user_defined_tangent` は `YUKinematic3D`（ntens=6）のみに存在する。
`YUKinematicPS` / `YUKinematic1D` は autograd 経路のみのため、Path A の strict 比較は 3D 専用とし、
PS / 1D は smoke テスト（収束確認 + 単調性）のみで対象とする。

## tolerance 方針と根拠

親 `tests/benchmarks/README.md` が定める基準値（stress atol=1e-6、tangent max_rel_err < 1e-5）から
本モデルは意図的に乖離している。乖離の根拠を以下に記録する。

### 構造的差分の原因

YU モデルの解析経路（`user_defined_return_mapping` + `user_defined_tangent`）と
autograd 経路の間には、**3 箇所の意図的な実装差分**がある。

#### (1) stagnation surface の g_flag（`yu_kinematic.py:63` vs `:154`）

| 経路 | 定義 |
|---|---|
| autograd (`update_state`) | `g_flag = smooth_heaviside(g_stag)` （連続・微分可能） |
| analytical (NR loop 内) | `g_flag = 1.0 if g_stag > 0.0 else 0.0` （不連続・ハード分岐） |

autograd で `smooth_heaviside` を使うのは JAX/autograd で微分可能にするためであり、
物理的な正解は不連続な指示関数（ハード分岐）である。
**どちらが「正しい」かは物理的に確定しない。**
この差分が state 変数 R / q / r に対して step あたり O(1e-2) の相対誤差を生む。

#### (2) calc_residual の C_k（`yu_kinematic.py:96` vs `:199`）

残差計算では両経路とも `smooth_heaviside(theta_max - (B-Y))` を使用しており、
**差分なし**。収束した解は同一の零点に対応するため、stress / state の一致精度が高い
（実測: stress max 2.9e-4 MPa）。

#### (3) _prepare_Rtheta の C_k（`yu_kinematic.py:224`）

| 経路 | 定義 |
|---|---|
| autograd (一致接線) | `smooth_heaviside(theta_max - (B-Y))` で微分 |
| analytical (`_prepare_Rtheta`) | `C_1 if B-Y > theta_max else C_2`（ハード if） |

これは **Jacobian（接線剛性）計算のみ**に影響する。残差は揃っているため応力解には影響しない。

### 実測値（`_diagnose.py` 実行結果）

`_diagnose.py` にて ground-truth-shared 方式（各 step で同一 `(deps, stress_n, state_n)` を
両 integrator に渡し、数値側で状態を進める）で測定した。

| シナリオ | plastic steps | max stress_err | max ddsdde_err | dist_to_75 相関 |
|---|---|---|---|---|
| uniaxial_monotonic | 27 / 50 | 2.9e-4 | 8.3e-4 | 無相関（均一分布） |
| small_amplitude_cyclic | 0 / 45 | — | — | — |
| uniaxial_cyclic | 192 / 200 | 2.5e-6 | 3.5e-2 | 無相関 |

`dist_to_75 = theta_max - (B-Y)` が ddsdde 誤差と相関しないことを確認。
最大誤差は遷移帯（`theta_max ≈ 75`）付近に集中せず、全 plastic step に均等に分布する
**Pattern B**（構造的差分）であった。
当初仮説の「遷移帯集中（Pattern A）」は棄却された。

### 採用 tolerance と根拠

| 量 | tolerance | 根拠 |
|---|---|---|
| stress | < 1e-3 | 実測 max 2.9e-4 に 3.4× のマージン。Y=360 MPa に対し相対 0.08% |
| state theta, beta, eps_eq, theta_max | rel < 1e-4 | 実測 max ~1e-4、smooth_heaviside の直接影響なし |
| state R, q, r | rel < 1e-2 | 実測 max ~1.2e-2、上記 (1) の g_flag 構造差に起因 |
| ddsdde | rel < 1e-1 | 実測 max 3.5e-2 に 2.9× のマージン。上記 (3) の Jacobian 近似に起因 |
| NR iter_diff | == 0 | 全シナリオで完全一致を実測。厳密 assert |

**ddsdde の 10% tolerance は解の正確さへの影響がないことに注意。**
ddsdde（一致接線剛性）は大域 FE Newton-Raphson の収束速度にのみ影響し、
収束した応力・状態変数の値には影響しない。局所 NR の反復数は完全一致（iter_diff=0）。

### 将来の改善候補（PR-D）

`_prepare_Rtheta`（`yu_kinematic.py:221`）の C_k を `smooth_heaviside` に揃えることで
ddsdde tolerance を 1e-1 → 1e-5 に戻せる可能性がある。
ただし (1) の g_flag 差は物理的な設計選択であり修正対象外。

## 診断スクリプト

`_diagnose.py`（アンダースコア prefix により pytest には collect されない）:

```bash
uv run python tests/benchmarks/yu_kinematic/_diagnose.py 2>&1 | tee /tmp/yu_diag.txt
```

各シナリオについて step ごとの `theta_max`、`dist_to_transition`、
stress / state / ddsdde 誤差を記録し、`ddsdde_err` 降順で上位 10 step を表示する。
tolerance 見直し時や B-Y パラメータ変更時の再評価に使用する。

## Path B tolerance と根拠

Path B は `PythonAnalyticalIntegrator`（`user_defined_return_mapping`）を主軸とし、
**同一のアルゴリズム（NR 50 iter + 内側 mu Newton 10 iter + ハード g_flag 分岐）**を
Fortran に忠実移植した `yu_kinematic_3d` と比較する。構造的差分がないため strict tolerance が適用される。

| 量 | tolerance | 根拠 |
|---|---|---|
| stress max_rel_err | < 1e-6 | 同一 NR 反復・倍精度演算の丸め誤差のみ |
| state max_rel_err | < 1e-6 | 同上 |
| ddsdde max_rel_err | < 1e-5 | 19×19 LU 逆行列の積算丸め誤差を含む |

Path A の構造的差分 tolerance（stress < 1e-3, ddsdde < 1e-1）は **Path B には適用されない**。
Path B の Fortran 移植元は解析経路であり、autograd 経路との差分は持たない。

## 実行方法

```bash
make test-benchmarks                 # slow 除外（fast CI 向け）
uv run pytest tests/benchmarks/yu_kinematic/ -v          # slow 含む全テスト
uv run pytest tests/benchmarks/yu_kinematic/ -m "not slow" -v  # 非 slow のみ
uv run pytest tests/benchmarks/yu_kinematic/ -m "fortran" -v   # Fortran テストのみ（.so 必須）
```

## ABAQUS 入力デック例

Fortran UMAT (`subroutine umat` in `fortran/yu_kinematic_3d.f90`) を ABAQUS で使用する場合の
入力デック例。`CONSTANTS=12` は `model.param_names` の 12 パラメータに対応。

```
*MATERIAL, NAME=YU_KINEMATIC
*USER MATERIAL, CONSTANTS=12
** E,        nu,    Y,    B,    C_1,   C_2,  Rsat,   k,   b,   h,      Ea,   xi
  206000., 0.3, 360., 435., 2000., 200., 255., 26., 66., 0.4, 159000., 61.
*DEPVAR
22
```

`*DEPVAR` に指定する 22 は STATEV スロット数（以下）:

| スロット | 変数 | 説明 |
|---|---|---|
| 1–6   | theta(1–6)  | 降伏面相対バックストレステンソル (physical shear) |
| 7–12  | beta(1–6)   | 境界面相対バックストレステンソル (physical shear) |
| 13    | R           | 境界面半径増分 |
| 14–19 | q(1–6)      | 停滞面中心 (physical shear) |
| 20    | r           | 停滞面半径 |
| 21    | eps_eq      | 相当塑性ひずみ |
| 22    | theta_max   | theta ノルムの履歴最大値 |

初期状態（弾性状態）はすべてのスロットにゼロを与えればよい。
非収束時は `PNEWDT = 0.5` が返り、ABAQUS に時間刻みの半減を要求する。
