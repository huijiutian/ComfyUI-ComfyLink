# 发版流程 — ComfyUI-ComfyLink

给维护者看的。**发布到 Comfy Registry 由「`pyproject.toml` 变化 + push main」触发**
(官方推荐方式):改好版本号推上去,`.github/workflows/publish.yml` 自动校验 + 发版。

> - 发布的版本号 = `pyproject.toml` 里的 `[project].version`。
> - **git tag 不参与触发**,只是里程碑记录 —— 发布成功之后再补打(见 1.5)。
> - Registry 更新后 ComfyUI-Manager 会自动同步(不需要给 Manager 仓库提 PR)。

### ⚠️ 这套触发方式的一个副作用(先知道,免得慌)

**改 `pyproject.toml` 的任何内容都会触发这条流水线**,哪怕只改了 description 或 Icon。
这时如果 version 没跟着变,Registry 会以「该版本已存在」拒绝(已发布的版本**不可覆盖**),
workflow 报红。**这不是故障,只是噪音。**

→ 所以:**改 pyproject 的非版本字段时,尽量和版本 bump 放进同一个提交**。
真的只想改个描述又不想 bump 版本,那就接受那次红叉(或者事后 re-run 也没用,忽略即可)。

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

**配好之后的第一次发布 = 0.2.1**(它已经 bump 但尚未发到 Registry):
只需空跑一次触发即可 —— 见 2. 常见问题「pyproject 没变但想重发」。

---

## 1. 每次发版

### 1.1 版本号四处同步(漏一处用户就收不到更新提醒)

| # | 文件 | 字段 | 仓库 |
|---|------|------|------|
| 1 | `pyproject.toml` | `[project].version` | plugin ← **改这个才触发发布** |
| 2 | `comfylink/version.py` | `__version__` | plugin |
| 3 | `ops/versions.json` | `plugin.latest` | plugin |
| 4 | `internal/api/versions.go` | `defaultPluginLatest` | **relay(另一个仓库)** |

- 1、2 不一致 → `tests/test_version.py` 会红,publish workflow 也会显式拦下(发布前)。
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

### 1.3 commit + push main(这一下触发发布)

```bash
git commit -am "chore(release): 插件版本 0.2.1 → 0.2.2"
git push origin main     # ← pyproject.toml 变了,发布流水线启动
```

### 1.4 看 workflow

<https://github.com/huijiutian/ComfyUI-ComfyLink/actions/workflows/publish.yml>

流程是 `verify` → `publish`:

1. **verify**:`pyproject.toml` 与 `comfylink/version.py` 版本号一致 → byte-compile →
   `unittest` → `ruff`。任一步红就**不会**发布。
2. **publish**:检查 secret 存在 → 官方 `Comfy-Org/publish-node-action@main`
   (内部 = `comfy node publish`)。

绿了以后去 <https://registry.comfy.org> 确认版本已经变成新的。

### 1.5 补打 tag(可选,纯记录)

发布成功之后再打,方便日后 `git diff v0.2.1..v0.2.2` 看某一版发了什么。
**打 tag 不会触发任何发布**,打错了删掉重打即可、无副作用。

```bash
git tag v0.2.2 && git push origin v0.2.2
```

---

## 2. 常见问题

**pyproject 没变但想(重新)发一次**
用 Actions 页面的 **Run workflow**(`workflow_dispatch`),ref 选 `main`。
——首次配好 secret 后发 0.2.1 走的就是这条;或者对已有的那次 run 点 **Re-run jobs**。

**Registry 当时挂了 / 发布步骤失败,代码没问题**
同上:**Re-run jobs** 或 **Run workflow**。

**改了 pyproject 的描述之类,版本没动,流水线报红**
预期行为(版本不可覆盖),忽略即可。下次记得把这类改动跟版本 bump 合进一个提交。

**同一个版本号重复发**
Registry 不接受覆盖已存在的版本。要重发就 bump 一个新版本号(按 1.1 改四处)。

**发布包里都装了什么**
由仓库根的 `.comfyignore` 决定(已排除 `tests/`、`.github/`、`ops/`、`RELEASING.md`、
`node.zip`、本地状态文件等)。
