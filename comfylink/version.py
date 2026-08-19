"""Single source of truth for the plugin version.

Keep this in sync with the `version` field in pyproject.toml (the Comfy
Registry reads pyproject; this constant feeds the runtime/panel).
"""

import subprocess
from pathlib import Path

__version__ = "0.2.4"

# CAPS —— 插件**自报的能力清单**,随 register 和每一拍心跳无条件上报(见
# relay.RelayClient)。App 用它决定要不要放出对应的入口。
#
# ⭐ 为什么是能力而不是版本号:`__version__` 历史上**没有随功能抬过**(采集 LoRA 那
# 批提交也没 bump),所以「版本号 >= X 就有这个功能」在这个仓库里是假命题。能力名由
# 代码本身携带,`git pull` + 重启 ComfyUI 就立刻生效,不需要重新配对。
#
#   "models" = 能按需重扫整个 models 目录并上传全量元信息(LoRA / checkpoint 等),
#              即 comfylink/loras.py 那套采集 + 心跳响应里 `loras_requested_at`
#              下行通道所服务的那件事。
#
#   "img2img" = 能把 App 传到 R2 的参考图取回来、塞进本机 ComfyUI 再回填节点输入,
#              即 comfylink/worker.py 的 `_stage_inputs` 所服务的那件事。
#              ⚠️ **`_stage_inputs` 从初始提交就有,但那不等于老插件都能用**:
#              2115a6f(2026-08-07)之前的 SSRF 守卫会**误伤我们自己的 R2** ——
#              在代理 fake-ip / NAT64 / DNS 污染的网络下,它把我们自己的下载 URL
#              当成内网地址拒掉(那笔提交的原话:on real users' machines rejected
#              our own R2),而取不到的失败模式是**静默跳过**(`if not url:
#              continue`)⇒ 用户拿到一张**没用上参考图**的图,还看不出哪里错了。
#              ⭐⭐ 而 2115a6f 前后 `__version__` **都是 0.2.2** —— 这条正是上面
#              那句「版本号 >= X 就有这个功能是假命题」的现成证据:App 只能靠这个
#              能力名放行,靠版本号会把一批注定静默出错的机器放进来。
#
# ⛔ **刻意不带版本号**(不是 "models.v3"),也**不要**从 loras.MANIFEST_SCHEMA 派生:
# caps 回答的是「支不支持这件事」。manifest schema 是插件↔App 之间**另一条**独立演进
# 的线,把它编进能力名等于每抬一次 schema 就让所有老 App 失能一次。
#
# ⚠️ 中继每拍**无条件覆盖**这个字段(空也写空),用户从新插件回退到老插件时能力要
# 立刻消失。所以插件这边也必须**无条件发**,不能写成「满足什么条件才带上」——
# 那会让能力在 App 里闪断。
__caps__ = ["models", "img2img"]


def _detect_commit() -> str:
    """Short git commit of the installed plugin (panel display, so the user can
    tell whether they `git pull`'d the latest). Best-effort + computed ONCE at
    import: any failure (no git, downloaded as zip, etc.) yields 'dev'."""
    try:
        root = Path(__file__).resolve().parent.parent  # plugin repo root
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "dev"
    except Exception:
        pass
    return "dev"


__commit__ = _detect_commit()
