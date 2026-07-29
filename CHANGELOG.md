# Changelog — ComfyUI-ComfyLink

自用版本记录（不面向 Comfy Registry 的展示页；Registry 读的是 `pyproject.toml`）。
版本号见 `pyproject.toml` 与 `comfylink/version.py`（**两处必须一致**，`tests/` 里有测试强制）。

版本号同时被 App/中继的更新提醒机制消费——发版要 bump 的完整文件清单见插件仓库 `ops/versions.json` 与 `app/docs/version-reminders.md`。

## [未发布] — 2026-07-29

⚠️ 发版前记得同步 bump `pyproject.toml` + `comfylink/version.py`（当前 `0.2.0`），并更新 `ops/versions.json` 的插件版本，否则用户端不会收到更新提醒。

### Added
- **面板柔和提醒插件有新版可更新**：不打断工作流，只在 ComfyLink 面板里给一个不刺眼的提示。

### Fixed
- **投递重试：结果/上传跨中继部署窗口不丢**。中继重新部署（Render 滚动更新）期间会有几十秒的窗口，此前落在窗口里的结果上报/产物上传会直接失败、任务卡住；现在带重试，跨过部署窗口后继续投递。
  > 这条与中继侧「device token 遇瞬时 DB 错误返回 503 而非 401」是同一类问题的两端——部署窗口不该让用户的任务受损。

---

## 更早

`0.2.0` 及之前未维护本文件，内容以 git log 为准。主要里程碑：

- **0.2.0** — 自报版本给中继 + 版本清单 `ops/versions.json`；工作流同步按账号勾选（目标账号多选 + 标记按账号隔离）
- 上架 **Comfy Registry**（publisher `comfylink`，Manager 自动同步，免 PR）
- 出图跟踪改为**版本无关的 `/history` + `/queue` 轮询**（原先靠 WS `executing{node:null}` 判完成，ComfyUI 升级改格式就卡 600s）；**精准取消**（pending 走 `/queue delete`、running 走 `/interrupt`，绝不误杀用户本地在跑的图）
- 产物并发上传 R2（限并发数）
