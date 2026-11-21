# 新しいモデルを足す

新しい構成則を、Python のモデルから検証済みの UMAT まで進める手順をまとめる。例は J2 + Voce 硬化 (飽和型の等方硬化) で、完成形が `examples/model_family/` にある。このディレクトリは、取り出せばそのまま自分のリポジトリの雛形になる。

各段の詳しい説明は別のページにある。

- フックの書き方: [モデルを作る](writing-models.md)
- 検査が何を測るか: [検証ワークフロー](verification.md)
- UMAT のビルドと比較: [Fortran UMAT クロス検証](fortran.md)

## どこまで進めるか

モデルは 3 つの段階のどこで止めてもよい。検査の内容は、モデルが宣言しているものから決まる。

| 段階 | 書くもの | 動く検査 |
|---|---|---|
| A. Python のモデル | `yield_function` と、状態の発展則 (`update_state` / `state_residual`) | 収束する、接線が差分と合う |
| B. 手書きの解法 | `user_defined_return_mapping` / `user_defined_tangent` (必要なら `@hand_block`) | 手書きの解が残差を満たす、自動微分と同じ履歴をたどる、手書きの接線が差分と合う |
| C. UMAT | Fortran のソースと `Umat` の宣言 | UMAT が Python と全ステップで一致する |

- **A だけで完結する。** リターンマッピングと整合接線は自動微分で解かれるので、計算・モデルの比較・パラメータ同定はこの段階でできる。
- **B は UMAT の下書きである。** 手で導いた代数を、自動微分を基準に確かめる。副産物として計算が速くなる (下の例で 1 ステップ 2.6 ms → 0.03 ms)。
- **C は B の移植である。** UMAT が比べられる相手は手書きの経路で、同じ代数を 2 つの言語で書いたものが一致するかを見る。

## 置き場所

1 つのモデルファミリを 1 つのディレクトリにまとめる。モデル本体、検証の宣言、UMAT のソース、テストが同じ場所にある。

```text
mymodels/                          ← リポジトリ
  pyproject.toml                   returnmap をリリースのタグに固定する
  src/mymodels/
    voce/                          ← 1 ファミリ
      __init__.py                  公開するクラス
      README.md                    定式化 (式、状態変数、パラメータ)
      model.py                     モデル本体
      verification.py              検証の宣言 (MODELS、UMATS)
      fortran/voce_3d.f90          UMAT のソース (C まで進むモデルだけ)
      tests/test_verification.py   共通の検査を回す
  fortran/                         コンパイルした UMAT の置き場 (ビルドすると出来る)
```

### 1 つのディレクトリに入れる範囲

ファミリは、1 つのモデルと、それを継承した変種である。

- **1 つのモデル** = 親クラス 1 つ (物理) と、その次元違いのクラス (`3D` / `PS` / `1D`)。
- **継承した変種は同じディレクトリに入れる。** 元のモデルに項を足したもの、段階を進めたものがこれに当たる (本体の `j2_damage/` は、損傷・crack closure・準片側の 3 段階を持つ)。
- **互いに独立なモデルは、系統が同じでも別のディレクトリにする。** 名前の頭をそろえて並べる (`yu_proj/`、`yu_continuous/` のように)。同じディレクトリに入れると、テストと UMAT のソースが混ざり、片方だけを動かせなくなる。
- **別のファミリのモデルを継承するときは、モデル本体がそのファミリを import してよい。** テストは、自分のファミリと公開の import パスだけを使う。

目安は `verification.py` である。そこに並ぶモデルが、同じ定式化のページ (`README.md`) 1 つで説明できる範囲に収まっていればよい。

returnmap は PyPI に無いので、`pyproject.toml` でタグに固定する。

```toml
[project]
dependencies = ["returnmap>=0.3.2"]

[project.optional-dependencies]
dev = ["pytest"]
fortran = ["meson", "ninja"]       # UMAT をビルドするとき

[tool.uv.sources]
returnmap = { git = "https://github.com/toruinaba/returnmap", tag = "v0.3.2" }

[tool.pytest.ini_options]
testpaths = ["src/mymodels"]
addopts = "--strict-markers --import-mode=importlib"
```

`fortran` / `slow` のマーカーと、ビルド先 (`fortran/`) を import できるようにする設定は、returnmap が配る pytest プラグインが行う。`conftest.py` に書くものは無い (例のディレクトリにある `conftest.py` は、returnmap をインストールせずに checkout から使うときのためのもの)。

**ファミリのテストは、公開の import パス (`returnmap.core` など) とそのファミリの中のものだけを import する。** ディレクトリごと別のリポジトリへ動かせるようにするためである。

## 1. モデルを書く (段階 A)

降伏関数と、状態の発展則を書く。Voce 硬化は `σ_y(p) = σ_y0 + Q (1 − exp(−b p))` である。

```python
# src/mymodels/voce/model.py
import autograd.numpy as anp
from returnmap.core import SCALAR, UNIAXIAL_1D, Explicit, MaterialModel, ParamMeta

class J2Voce(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "Q", "b"]
    param_meta = {
        "E": ParamMeta(doc="Young's modulus", unit="MPa", default=210_000.0,
                       bounds=(1_000.0, 1_000_000.0)),
        ...
    }
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_stress(self, ep):
        return self.sigma_y0 + self.Q * (1.0 - anp.exp(-self.b * ep))

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - self.yield_stress(state["ep"])

    def update_state(self, new, dlambda, inc):
        return [self.ep(inc.state_n["ep"] + dlambda)]

class J2Voce1D(J2Voce):
    dimension = UNIAXIAL_1D
```

- **次元ごとの式は書かない。** `vonmises` などの演算子が次元を吸収するので、物理は親クラスに 1 度書く。次元を固定したクラスは `dimension` を書くだけである。
- **`param_meta` は任意。** 書いておくと `J2Voce.at_defaults()` で目安の値のモデルが作れ、検証の宣言、調整用のノートブック、service の入力フォームがそれを読む。

## 2. 検証を宣言して回す

検査にかけるモデルを `verification.py` に並べる。

```python
# src/mymodels/voce/verification.py
from mymodels.voce.model import J2Voce1D

MODELS = [
    J2Voce1D.at_defaults(),
]
```

テストは、どのファミリでも同じ数行である。

```python
# src/mymodels/voce/tests/test_verification.py
import pytest
from returnmap.verification import check_model
from mymodels.voce.verification import MODELS

@pytest.mark.parametrize("model", MODELS, ids=lambda model: type(model).__name__)
def test_passes_every_check_that_applies(model):
    report = check_model(model)
    assert report.passed, report.summary()
```

```bash
uv sync --extra dev
uv run pytest
```

`check_model` は標準の載荷履歴 (単軸の載荷・除荷・逆載荷と、向きを変える多軸の履歴) でモデルを走らせ、当てはまる検査を全部回す。段階 A のモデルの報告は次のとおりで、`—` は宣言していないものである。不合格には数えない。

```text
J2Voce1D: every check that applies passes
  ok   converges                 35 steps over 1 histories, 24 plastic
  ok   tangent                   7.2e-10 (tol 1e-05) at 4 plastic steps
  —    hand blocks               no @hand_block on the model
  —    hand-written solution     no user_defined_return_mapping on the model
  —    hand-written vs autodiff  no user_defined_return_mapping on the model
  —    hand-written tangent      no user_defined_return_mapping on the model
  —    UMAT                      no UMAT declared for J2Voce1D in mymodels.voce.verification
  histories: uniaxial (24 plastic steps)
```

標準の履歴は降伏ひずみの 5 倍までしか行かない。それでは届かない挙動を持つモデル (大きなひずみからの反転で効く硬化則など) は、`verification.py` に `histories(model)` を書いて履歴を足す。弾性域の無いモデルは、履歴の尺度にするひずみを渡す ([検証ワークフロー](verification.md) の `check_model`)。

この段階で、モデルは使える。試験データへの合わせ込みは [チュートリアル](tutorial.md)、ほかのモデルとの比較は [モデルを比べる](comparing.md) を参照。

## 3. 手書きの解法を足す (段階 B)

UMAT に書く代数を、まず Python で書く。J2 のリターンは半径方向なので、1 ステップはスカラーの方程式 `g(Δλ) = q − 3μ Δλ − σ_y(p_n + Δλ) = 0` になる (`q` は試行応力の Mises 応力)。この恒等式は直応力 3 成分が揃っている次元でしか成り立たないので、クラスを 3D に固定する。

```python
class J2Voce3D(J2Voce):
    dimension = SOLID_3D
    MAX_ITER = 25
    TOL = 1.0e-12

    def user_defined_return_mapping(self, inc):
        mu = self.E / (2.0 * (1.0 + self.nu))
        stress_trial = inc.trial["stress"]
        ep_n = inc.state_n["ep"]
        s_trial = self.dev(stress_trial)
        q = self.vonmises(stress_trial)

        dlambda, residual_history, converged = 0.0, [], False
        for _ in range(self.MAX_ITER + 1):
            g = q - 3.0 * mu * dlambda - self.yield_stress(ep_n + dlambda)
            residual_history.append(float(abs(g)))
            if abs(g) < self.TOL * q:
                converged = True
                break
            dlambda = dlambda + g / (3.0 * mu + self.hardening_modulus(ep_n + dlambda))

        stress = stress_trial - (3.0 * mu * dlambda / q) * s_trial
        return ReturnMappingResult(
            stress=stress, state={"stress": stress, "ep": ep_n + dlambda},
            dlambda=anp.asarray(dlambda), n_iterations=len(residual_history) - 1,
            residual_history=residual_history, converged=converged,
        )

    def user_defined_tangent(self, new, dlambda, inc):
        ...   # D = I_vol C + θ I_dev C − β s_trial ⊗ s_trial
```

`MODELS` に `J2Voce3D.at_defaults()` を足すと、手書きの経路の検査が動く。

```text
J2Voce3D: every check that applies passes
  ok   converges                 59 steps over 2 histories, 47 plastic
  ok   tangent                   4.6e-10 (tol 1e-05) at 8 plastic steps
  —    hand blocks               no @hand_block on the model
  ok   hand-written solution     4.9e-13 (tol 1e-08) over 3 rows at 47 plastic steps
  ok   hand-written vs autodiff  stress 3.3e-12, state 4.2e-12 (tol 1e-06), tangent 4.8e-12 (tol 1e-05) over 2 histories
  ok   hand-written tangent      4.6e-10 (tol 1e-05) at 8 plastic steps
  —    UMAT                      no UMAT declared for J2Voce3D in mymodels.voce.verification
  histories: uniaxial (24 plastic steps), multiaxial (23 plastic steps)
```

| 行 | 確かめていること |
|---|---|
| hand-written solution | 手書きの解を、フレームワークの残差 (自動微分側の方程式) に代入すると 0 になる |
| hand-written vs autodiff | 2 つの経路を同じ履歴で進めて、応力・状態・接線が全ステップで一致する |
| hand-written tangent | 手書きの接線が、応力更新の中心差分と合う |

書くときの決まりは 3 つある。

- **`converged` は必ず自分で報告する。** 既定値は無い。反復の上限で抜けたときに成功と報告しないためである。
- **Δλ < 0 と非有限の値は、`converged=True` と報告しても非収束として扱われる。**
- **基準は自動微分である。** 2 つが合わないときは、まず手導出を疑う。

不合格の行の中身は、報告から取り出せる。

```python
report = check_model(model)
print(report["hand-written tangent"].reports["multiaxial[1]"].summary())
```

手書きの NR がヤコビアンを持つモデル (未知数が複数あるもの) は、ブロックを `@hand_block(row, col)` で宣言すると、自動微分のヤコビアンとブロックごとに照合される。意図して省いた項があるときは、許容値を緩めずに、宣言に書く (`approx=`、`approx_when=`、接線は `@hand_tangent`)。書き方は [検証ワークフロー](verification.md) を参照。

## 4. UMAT にする (段階 C)

手書きの経路を Fortran に移す。ソースはファミリの `fortran/` に置き、2 つのサブルーチンに分ける。

- **中核** (`voce_3d`): 1 ステップの構成則の更新。f2py でビルドして Python と比べるのはこちらである。
- **入口** (`umat`): ABAQUS が呼ぶサブルーチン。`PROPS` と `STATEV` をほどいて中核を呼ぶだけにする。

中核の引数は、パラメータ (`param_names` の順)、応力、状態変数 (宣言の順)、ひずみ増分の順に並べ、出力は応力、状態変数、接線の順にする。こうしておけば、呼び出し方を宣言に書く必要が無い。

```fortran
subroutine voce_3d(E, nu, sigma_y0, Q, b, &
                   stress_in, ep_in, dstran, &
                   stress_out, ep_out, ddsdde, converged)
    double precision, intent(in)  :: E, nu, sigma_y0, Q, b
    double precision, intent(in)  :: stress_in(6), ep_in, dstran(6)
    double precision, intent(out) :: stress_out(6), ep_out, ddsdde(6,6)
    integer,          intent(out) :: converged     ! 1 = converged, 0 = not
```

UMAT を `verification.py` に宣言する。

```python
from returnmap.verification import Umat

def _umat_failure(ret):
    """voce_3d は (stress, ep, ddsdde, converged) を返す。"""
    return None if int(ret[3]) == 1 else "the UMAT's Newton iteration did not converge"

UMATS = [
    Umat(J2Voce3D, module="voce_3d", sources=["fortran/voce_3d.f90"],
         integrator_options={"parse_umat_failure": _umat_failure}),
]
```

- **`sources` はモデルのモジュールがあるディレクトリからの相対パス。**
- **UMAT が自分の収束フラグを返すなら、`parse_umat_failure` で読ませる。** 指定しないと、UMAT が解けなかったステップの値が解として比べられる。
- **`PROPS` の順が `param_names` と違う、または一部しか受け取らない UMAT は `props=[...]` で宣言する。** LAPACK などが要るなら `libraries=["lapack", "blas"]`。

ビルドは宣言を読んで行われる。ABAQUS のユーティリティ (`SINV` など) の代わりになるスタブは returnmap が持っていて、一緒にコンパイルされる。

```bash
uv sync --extra dev --extra fortran           # gfortran が要る
uv run returnmap build --family mymodels      # 宣言された UMAT を全部ビルドする
uv run pytest
```

ビルドすると、報告の UMAT の行が動く。標準の履歴の全ステップで、応力・状態・接線を Python のモデルと比べる。

```text
  ok   UMAT                      stress 2.7e-16, state 5.2e-16 (tol 1e-06), tangent 4.0e-16 (tol 1e-05) over 2 histories (voce_3d)
```

ビルドしていない環境では、この行は `not built` になり、不合格には数えない。gfortran の無い環境でも検査が通るようにするためである。UMAT の検査を必ず走らせたい所では、`fortran` マーカーのテストで行の状態を確かめる。

```python
@pytest.mark.fortran
@pytest.mark.parametrize("umat", UMATS, ids=lambda umat: umat.module)
def test_the_umat_agrees_with_the_python_model(umat):
    pytest.importorskip(umat.module, reason=f"{umat.module} is not built")
    report = check_model(umat.model.at_defaults())
    assert report["UMAT"].status == "passed", report.summary()
```

gfortran の無いホストでは Docker を使う。returnmap の `Dockerfile` が雛形になる。

## 5. 固有のテストを足す

共通の検査で足りないもの (特定のパラメータでの回帰、文献値との比較など) は、ファミリの `tests/` に足す。名前の決まりは次のとおりである。

| 名前 | 中身 |
|---|---|
| `test_verification.py` | 共通の検査だけ。固有のテストをここに書かない |
| `test_model.py` | モデルの関数ごとの確認 (integrator を使わない) |
| `test_<確かめること>.py` | そのモデルに固有の確認 |

共通の検査がまだ通らないモデルは、分かっている理由を付けて `pytest.mark.xfail(strict=True, reason=...)` にしておく。直ったときに気づける。

## returnmap 本体に入れるとき

ファミリのディレクトリをそのまま `src/returnmap/models/<family>/` に移す。テストが公開の import パスだけを使っていれば、変えるのは次の箇所である。

1. ファミリの中の import を `returnmap.models.<family>` に変える。
2. `returnmap/models/__init__.py` の `__all__` にクラスを足す。
3. 全パラメータに `param_meta` を書く (組み込みモデルは、目安の値で計算が通ることをテストで確かめている)。
4. ファミリの `README.md` に定式化を書き、`docs/guide/models/<family>.md` から取り込む。
5. [組み込みモデルと次元選択](models.md) の表と `CHANGELOG.md` に足す。

## 確認の一覧

| 段階 | 済んだと言える状態 |
|---|---|
| A | `check_model` の `converges` と `tangent` が `ok` |
| B | `hand-written solution` / `hand-written vs autodiff` / `hand-written tangent` が `ok`。近似は宣言に書いてある |
| C | ビルドした環境で `UMAT` の行が `ok`。UMAT の収束フラグを `parse_umat_failure` で読ませている |
