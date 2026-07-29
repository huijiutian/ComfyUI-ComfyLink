# 发版流程 — ComfyUI-ComfyLink

给维护者看的。**发布到 Comfy Registry 现在由 tag 触发**:推一个 `v*` tag,
`.github/workflows/publish.yml` 自动校验 + 发版。

> 只 `git push` 到 main **不会**发到 Registry。Registry 上的版本只由 tag 决定。
> Registry 更新后 ComfyUI-Manager 会自动同步(不需要给 Manager 仓库提 PR)。

---

## 0. 一次性配置(只做一次,没做的话 workflow 必失败)

### 0.1 申请 Registry token

1. 打开 <https://registry.comfy.org/nodes>,用发布者账号登录。
2. 切到 publisher **`comfylink`**(页面上的 publisher 选择器)。
3. 在该 publisher 下 **创建 API Key / Personal Access Token**,复制生成的字符串。
   **只显示一次**,离开页面就看不到了。

### 0.2 存进 GitHub 仓库 secret

1. 打开 <https://github.com/huijiutian/ComfyUI-ComfyLink/settings/secrets/actions>
   (即仓库 → **Settings → Secrets and variables → Actions**)。
2. **New repository secret**。
3. Name 必须**一字不差**是:

   ```
   REGISTRY_ACCESS_TOKEN
   ```

4. Secret 填 0.1 复制的 token → **Add secret**。

> token 绝不写进仓库任何文件。workflow 只从 `secrets.REGISTRY_ACCESS_TOKEN` 读。
> 万一泄漏:去 registry.comfy.org 撤销旧 key,重新生成并覆盖这个 secret。

---

## 1. 每次发版

### 1.1 版本号四处同步(漏一处用户就收不到更新提醒)

| # | 文件 | 字段 | 仓库 |
|---|------|------|------|
| 1 | `pyproject.toml` | `[project].version` | plugin |
| 2 | `comfylink/version.py` | `__version__` | plugin |
| 3 | `ops/versions.json` | `plugin.latest` | plugin |
| 4 | `internal/api/versions.go` | `defaultPluginLatest` | **relay(另一个仓库)** |

- 1、2 不一致 → `tests/test_version.py` 会红,publish workflow 也会显式拦下。
- **3、4 没有任何自动校验,只能靠人**:
  - 漏 3 → App 端「有新版可更新」提醒**根本不出现**(App 读的就是这个 raw 文件)。
  - 漏 4 → 连不上 GitHub、走中继 `GET /v1/versions` 兜底的用户会被喂**过期版本号**。
  - 改 4 需要在 relay 仓库单独 commit + push(**push main = 触发 Render 生产部署**,注意时机)。

顺手更新 `CHANGELOG.md`。

### 1.2 本地跑一遍绿

```bash
python -m py_compile __init__.py comfylink/*.py
python -m unittest discover -s tests
ruff check .
```

### 1.3 commit + 打 tag + push

```bash
git commit -am "chore(release): 插件版本 0.2.1 → 0.2.2"
git push origin main

git tag v0.2.2          # 必须是 v + pyproject 里的版本号,一字不差
git push origin v0.2.2  # ← 这一下才触发发布
```

### 1.4 看 workflow

<https://github.com/huijiutian/ComfyUI-ComfyLink/actions/workflows/publish.yml>

流程是 `verify` → `publish`:

1. **verify**:必须跑在 tag 上 → tag 版本号 vs `pyproject.toml` vs `comfylink/version.py`
   三者一致 → byte-compile → `unittest` → `ruff`。任一步红就**不会**发布。
2. **publish**:检查 secret 存在 → 官方 `Comfy-Org/publish-node-action@main`
   (内部 = `comfy node publish`)。

绿了以后去 <https://registry.comfy.org> 确认版本已经变成新的。

---

## 2. 常见问题

**tag 打错了 / 版本号对不上被拦下**

```bash
git tag -d v0.2.2
git push origin :refs/tags/v0.2.2   # 删远端 tag
# 修好版本号并 commit 之后重新打
git tag v0.2.2 && git push origin v0.2.2
```

**Registry 当时挂了 / 发布步骤失败,代码没问题**
在 Actions 页面对那次 run 点 **Re-run jobs**;或者用 **Run workflow**
(`workflow_dispatch`)手动跑,**ref 必须选那个 v\* tag**(选分支会被 guard 拒掉)。

**同一个版本号重复发**
Registry 不接受覆盖已存在的版本。要重发就 bump 一个新版本号(1.1 四处 + 新 tag)。

**发布包里都装了什么**
由仓库根的 `.comfyignore` 决定(已排除 `tests/`、`.github/`、`ops/`、本地状态文件等)。
