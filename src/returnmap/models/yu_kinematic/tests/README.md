# yu_kinematic のテスト

Tests of the Yoshida-Uemori two-surface + stagnation-surface
kinematic hardening model (`YUKinematic3D / YUKinematicPS / YUKinematic1D`).

`YUKinematic3D` と `YUKinematicPS` は、autograd 経路に加えて手書きの経路
(`user_defined_return_mapping` / `user_defined_tangent`、手導出ヤコビアン) を持つ。
`YUKinematic1D` は autograd 経路のみ。

## ファイル

| ファイル | 内容 |
|---|---|
| `test_verification.py` | 共通の検査 (`check_model`) を 3D / PS / 1D に回す。3D と PS の手導出は、停留曲面のゲートが切り替わるステップだけ近似と宣言してある (冒頭の docstring)。UMAT をビルドしてあれば、同じ履歴で Python と突き合わせる |
| `test_model.py` | モデル単体: 親クラスと次元別クラスの一致、停留曲面の更新、残差とヤコビアンの部品 |
| `test_analytical_vs_numerical.py` | 3D: 手書きの経路 vs autograd NR (応力・状態・接線・反復数)。1D は収束と単調性の smoke |
| `test_jacobian_fd.py` | 3D: `calc_jacobian` vs 差分 (停留曲面の状態を毎回再計算した全微分) |
| `test_plane_stress_p.py` | PS: P 計量の規約、3D・1D との一致、手導出ブロック (`check_hand_blocks`)、手書きの解 (`check_solution`) |
| `test_numerical_vs_fortran.py` | 3D: Python vs Fortran (`yu_kinematic_3d`、要ビルド) |
| `test_ps_vs_fortran.py` | PS: Python vs Fortran (`yu_kinematic_ps`、要ビルド) |
| `_diagnose.py` | 診断スクリプト (pytest の対象外)。下記 |

## 許容値

許容値とその根拠 (実測値、過去に隠れていた不具合) は、**各テストモジュールの docstring が正**である。
ここには写しを置かない。以前この README にあった表は、2 経路の停留曲面ゲートが揃う前の
ものだった (現在は応力・状態が機械精度で一致する)。

2 つの実装を比べるテストの基準値 (`docs/development.md` の「モデルのテスト」) から外れているものは、モジュールの docstring に理由を書く。

## 診断スクリプト

`_diagnose.py` は、各シナリオについて step ごとの `theta_max`、遷移点 (`B − Y`) までの距離、
stress / state / ddsdde の誤差を出し、`ddsdde` 誤差の大きい順に上位 10 step を表示する。
許容値を見直すときや、`B − Y` 付近のパラメータを変えたときに使う。

```bash
uv run python src/returnmap/models/yu_kinematic/tests/_diagnose.py 2>&1 | tee /tmp/yu_diag.txt
```

## 実行方法

```bash
uv run pytest src/returnmap/models/yu_kinematic -m "not fortran" -v             # Fortran 以外 (slow を含む)
uv run pytest src/returnmap/models/yu_kinematic/tests/ -m "not fortran and not slow" -v    # 速いものだけ
make docker-build && make docker-test-yu                                     # Fortran 比較 (ホストに gfortran が無い場合)
```
