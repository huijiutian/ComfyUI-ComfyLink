#!/usr/bin/env bash
#
# ComfyUI-ComfyLink 插件 —— 发版前检查(preflight)
#
# 只读:不改任何文件、不 commit、不 push、不发 Registry。
#
# 用法:
#   ./tools/release_preflight.sh            # 全量
#   ./tools/release_preflight.sh --offline  # 跳过 git fetch
#   ./tools/release_preflight.sh --no-test  # 跳过 unittest
#
# 插件和 App 的时序不一样:插件的 ops/versions.json 里 plugin.latest 指向 GitHub 仓库,
# push 完用户 `git pull` 就拿得到 —— 所以版本号四处**同一个提交里一起改、一起 push** 即可,
# 不用像 App 那样等上架。但 Comfy Registry 是**另一条发布通道**,push 不等于发布,见末尾提示。
#
# 完整流程见 ../app/docs/release-runbook.md。
set -euo pipefail

DO_TEST=1; DO_FETCH=1
for arg in "$@"; do
  case "$arg" in
    --no-test) DO_TEST=0 ;;
    --offline) DO_FETCH=0 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg(支持 --no-test / --offline)"; exit 2 ;;
  esac
done

PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$PLUGIN_ROOT/.." && pwd)"
RELAY_DIR="${RELAY_DIR:-$WS/relay}"
cd "$PLUGIN_ROOT"

LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/comfylink-plugin-preflight.XXXXXX")"

PASSN=0; FAILN=0; WARNN=0
ok()   { printf '  \033[32m✅\033[0m %s\n' "$*"; PASSN=$((PASSN+1)); }
bad()  { printf '  \033[31m❌\033[0m %s\n' "$*"; FAILN=$((FAILN+1)); }
warn() { printf '  \033[33m⚠️ \033[0m %s\n' "$*"; WARNN=$((WARNN+1)); }
info() { printf '     \033[2m%s\033[0m\n' "$*"; }
fix()  { printf '     \033[36m↳ 修复:%s\033[0m\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# 只用真实退出码判断成败(绝不 `cmd | tail && ...` —— 管道会把退出码换成 tail 的)。
run_logged() { local name="$1"; shift; if "$@" >"$LOGDIR/$name.log" 2>&1; then return 0; else return 1; fi; }
show_tail() { printf '\033[2m'; tail -n "${2:-30}" "$LOGDIR/$1.log"; printf '\033[0m'; }

# ── 发版审查门禁 ──────────────────────────────────────────────────────────────
# 复制在三个 preflight 里(三个独立仓库,没法共享库)。改一处记得改另两处。
#
# 为什么是硬门禁而不是提醒:preflight 的其它检查查的是「东西齐不齐」,审查查的是
# 「改动安不安全」——后者没法自动化,只能靠人。脚本唯一能做的,是要求那次思考
# 留下一份可检查的产物。在这套门禁建立之前,兼容性核查靠"碰巧想起来"。
#
# 用法:gate_check <scope> <target-version> <review-dir> <review-dir-所在的 git 仓库>
gate_field() {  # $1=文件 $2=字段名 → 回显字段值(去掉 <!-- --> 注释和首尾空白)
  grep -m1 -E "^- \*\*$2\*\*:" "$1" 2>/dev/null \
    | sed -E "s/^- \*\*$2\*\*:[[:space:]]*//; s/<!--.*-->//; s/[[:space:]]+$//" || true
}
gate_check() {
  local scope="$1" target="$2" dir="$3" repo="$4"
  local f found="" fscope ftarget verdict unchecked rel
  if [ ! -d "$dir" ]; then
    bad "找不到审查归档目录 $dir"
    fix "mkdir -p $dir && cp <app>/docs/release-review-template.md $dir/$(date +%F)-$scope-v$target.md"
    return
  fi
  for f in "$dir"/*.md; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in README.md) continue ;; esac
    fscope="$(gate_field "$f" 发版范围)"
    ftarget="$(gate_field "$f" 目标版本)"
    if { [ "$fscope" = "$scope" ] || [ "$fscope" = "all" ]; } && [ "$ftarget" = "$target" ]; then
      found="$f"; break
    fi
  done
  if [ -z "$found" ]; then
    bad "没有找到本次发版($scope / $target)的审查记录 —— 【不允许发版】"
    info "审查查的是「改动安不安全」(删号函数漏表了吗?老客户端解析新响应会不会崩?),"
    info "这是 preflight 其它检查覆盖不到的部分,只能人过脑子,但必须留下产物。"
    fix "cp <app>/docs/release-review-template.md $dir/$(date +%F)-$scope-v$target.md"
    fix "逐条填完(不留空的 - [ ]),然后 git add + commit,再回来跑本脚本"
    return
  fi
  rel="$(basename "$found")"
  ok "找到审查记录:$rel"

  unchecked="$(grep -cE '^[[:space:]]*- \[ \]' "$found" || true)"
  if [ "${unchecked:-0}" -eq 0 ]; then
    ok "审查清单已全部勾选(没有遗留的 - [ ])"
  else
    bad "审查记录里还有 $unchecked 项没勾 —— 审查没做完,【不允许发版】"
    grep -nE '^[[:space:]]*- \[ \]' "$found" | head -10 | sed 's/^/       /'
    fix "不适用的项也要勾上并在结论栏写一句为什么 N/A —— 留空等于没想"
  fi

  verdict="$(gate_field "$found" 最终结论)"
  case "$verdict" in
    通过)
      ok "最终结论:通过" ;;
    有条件通过)
      ok "最终结论:有条件通过 —— 条件如下,确认你认了这些条件再继续:"
      # 条件正文可能很长,截断显示;要看全的去翻文件本身。
      gate_field "$found" 条件与跟进项 | cut -c1-150 | sed 's/^/       /'
      sed -n '/^## 3\./,$p' "$found" | grep -E '^\| *[0-9]+ *\|' \
        | awk -F'|' '{p=$3; gsub(/^[ \t]+|[ \t]+$/,"",p); printf "       · %.110s\n", p}' | head -5 || true
      info "(完整条件见 $rel §3)" ;;
    不通过)
      bad "最终结论:不通过 —— 【拒绝发版】。先把审查里指出的问题解决掉。" ;;
    "")
      bad "审查记录的「最终结论」是空的 —— 【不允许发版】"
      fix "在 $rel 里把 - **最终结论**: 填成 通过 / 有条件通过 / 不通过 之一" ;;
    *)
      bad "看不懂的最终结论「$verdict」—— 只接受 通过 / 有条件通过 / 不通过" ;;
  esac

  # 返工记录:真机测试打回后要在【同一份】文件里追加,每条都必须有「复核结论」。
  # 有返工却没复核 = 代码改了、审查没跟上,比没审查更危险(看着有凭证,其实过期了)。
  local badrows
  badrows="$(sed -n '/^## 4\./,$p' "$found" | grep -E '^\| *[0-9]+ *\|' \
    | awk -F'|' '{v=$(NF-1); gsub(/^[ \t]+|[ \t]+$/,"",v);
                  if (v=="" || v=="—" || v=="-" || v=="待填") print}' || true)"
  if [ -n "$badrows" ]; then
    bad "返工记录里有条目没填「复核结论」—— 审查没跟上代码,【不允许发版】"
    printf '%s\n' "$badrows" | sed 's/^/       /'
    fix "在 $rel 的§4 返工记录里补上这次返工重新核对的结论"
  fi

  # 未提交的审查记录 = 没归档。git 里没有它,就等于这次发版没有凭证。
  if git -C "$repo" ls-files --error-unmatch "$found" >/dev/null 2>&1; then
    if git -C "$repo" diff --quiet HEAD -- "$found"; then
      ok "审查记录已提交进 git(已归档)"
    else
      bad "审查记录有未提交的改动 —— 归档的必须是最终版,【不允许发版】"
      fix "git -C $repo add $found && git -C $repo commit"
    fi
  else
    bad "审查记录还没加进 git —— 未提交 = 没归档,【不允许发版】"
    fix "git -C $repo add $found && git -C $repo commit"
  fi
}

APP_DIR="${APP_DIR:-$WS/app}"
# 版本号提前解析(门禁要用它匹配审查记录;第 3 步会再做四处一致性核对)。
V_PY_EARLY="$(grep -oE '__version__ *= *"[^"]*"' comfylink/version.py | grep -oE '"[^"]*"' | tr -d '"' || true)"

printf '\n\033[1m═══ ComfyUI-ComfyLink 插件 · 发版前检查 ═══\033[0m\n'
printf '     仓库 %s\n     日志 %s\n' "$PLUGIN_ROOT" "$LOGDIR"

# ── 0. 发版审查门禁 ───────────────────────────────────────────────────────────
step "0. 发版审查门禁(硬门禁 —— 没有审查记录就不允许发版)"
info "详见 ../app/docs/release-runbook.md §2「发版前审查」。模板 ../app/docs/release-review-template.md。"
info "审查记录统一归档在 app 仓库的 docs/release-reviews/(三件套共用一个归档目录)。"
gate_check plugin "$V_PY_EARLY" "$APP_DIR/docs/release-reviews" "$APP_DIR"


# ── 1. Python 解释器 ─────────────────────────────────────────────────────────
step "1. Python 解释器(需要 3.10+;CI 用 3.11)"
info '本机 /usr/bin/python3 是 Xcode 自带的 3.9,跑测试会在 test_subscriptions.py 的'
info 'PEP-604 写法(str | None)上直接 ImportError —— 那不是测试挂了,是解释器太老。'
# 用字符串而不是数组:macOS 自带 bash 3.2 下,空数组 + set -u 会报 unbound variable。
PY_CMD=""
if [ -n "${PYTHON:-}" ]; then
  PY_CMD="$PYTHON"
else
  for cand in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PY_CMD="$cand"; break; fi
  done
fi
if [ -z "$PY_CMD" ] && command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY_CMD="python3"
  fi
fi
if [ -z "$PY_CMD" ] && command -v uv >/dev/null 2>&1; then
  # uv 会按需拉 3.11 并在临时环境里带上 aiohttp —— 和 CI 一致(测真实 import 路径,
  # 而不是测试里那个 aiohttp stub 兜底)。
  PY_CMD="uv run --quiet --python 3.11 --with aiohttp python"
fi
if [ -z "$PY_CMD" ]; then
  bad "找不到 3.10+ 的 Python,也没有 uv"
  fix "brew install python@3.11  或  brew install uv;也可以 PYTHON=/path/to/python3.11 再跑一次"
  DO_TEST=0
else
  PYVER="$($PY_CMD -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo '?')"
  ok "解释器:$PY_CMD(Python $PYVER)"
fi

# ── 2. 仓库状态 ──────────────────────────────────────────────────────────────
step "2. 仓库状态"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] && ok "分支 = main" || warn "当前分支是 $BRANCH,不是 main"
DIRTY="$(git status --porcelain)"
if [ -z "$DIRTY" ]; then ok "工作区干净"
else
  bad "工作区有未提交改动"
  info "$(printf '%s' "$DIRTY" | head -10)"
fi
if [ "$DO_FETCH" = "1" ]; then
  if run_logged git-fetch git fetch --quiet origin; then ok "已 fetch origin"; else warn "git fetch 失败(离线?)"; fi
fi
if git rev-parse --verify --quiet "origin/$BRANCH" >/dev/null; then
  AHEAD="$(git rev-list --count "origin/$BRANCH..HEAD")"
  BEHIND="$(git rev-list --count "HEAD..origin/$BRANCH")"
  if [ "$AHEAD" = "0" ] && [ "$BEHIND" = "0" ]; then
    ok "与 origin/$BRANCH 一致(已推送)"
  else
    if [ "$AHEAD" != "0" ];  then warn "本地领先 $AHEAD 个提交 —— push 后用户 git pull 才拿得到"; fi
    if [ "$BEHIND" != "0" ]; then bad "本地落后 $BEHIND 个提交 —— 先 git pull"; fi
  fi
fi
ok "commit $(git rev-parse --short HEAD)(插件会把它上报给中继,App 里能看到)"

# ── 3. 版本号(四处必须一致)─────────────────────────────────────────────────
step "3. 版本号(四处必须一致)"
V_PY="$(grep -oE '__version__ *= *"[^"]*"' comfylink/version.py | grep -oE '"[^"]*"' | tr -d '"' || true)"
V_TOML="$(grep -oE '^version *= *"[^"]*"' pyproject.toml | grep -oE '"[^"]*"' | tr -d '"' || true)"
V_JSON="$(python3 -c 'import json,sys;print(json.load(open("ops/versions.json"))["plugin"]["latest"])' 2>/dev/null || true)"
V_GO=""
VGO_PATH="$RELAY_DIR/internal/api/versions.go"
if [ -f "$VGO_PATH" ]; then
  V_GO="$(grep -oE 'defaultPluginLatest *= *"[^"]*"' "$VGO_PATH" | grep -oE '"[^"]*"' | tr -d '"' || true)"
else
  warn "找不到 $VGO_PATH(用 RELAY_DIR= 指定中继仓库),中继兜底常量无法核对"
fi
info "comfylink/version.py    __version__          = ${V_PY:-?}"
info "pyproject.toml          version              = ${V_TOML:-?}"
info "ops/versions.json       plugin.latest        = ${V_JSON:-?}"
info "relay .../versions.go   defaultPluginLatest  = ${V_GO:-<未检查>}"

if [ -n "$V_PY" ] && [ "$V_PY" = "$V_TOML" ]; then
  ok "version.py 与 pyproject.toml 一致($V_PY)"
else
  bad "version.py($V_PY)与 pyproject.toml($V_TOML)不一致"
  info "Comfy Registry 读的是 pyproject,面板/上报读的是 version.py —— 不一致会让 Registry 上的版本和用户看到的版本对不上。"
  info "tests/ 里有测试守着这两处,所以下面的 unittest 也会红。"
fi
if [ -n "$V_JSON" ] && [ "$V_JSON" = "$V_PY" ]; then
  ok "ops/versions.json 的 plugin.latest 已同步"
else
  bad "ops/versions.json plugin.latest($V_JSON)≠ 代码版本($V_PY)"
  info "后果:用户的插件已经是新版了,App 却还按旧的 latest 比对 —— 要么该提醒的不提醒,要么已更新的还被提醒。"
  fix "把 ops/versions.json 的 plugin.latest 改成 $V_PY"
fi
if [ -n "$V_GO" ]; then
  if [ "$V_GO" = "$V_PY" ]; then ok "中继兜底常量 defaultPluginLatest 已同步"
  else
    bad "中继 defaultPluginLatest($V_GO)≠ 代码版本($V_PY)"
    info "后果:连不上 raw.githubusercontent.com 的用户(主要是国内)走中继兜底源,永远被喂过期版本号。"
    fix "改 $VGO_PATH 的 defaultPluginLatest = \"$V_PY\";注意中继 push main = 触发 Render 生产部署"
  fi
fi
if [ -f CHANGELOG.md ] && [ -n "$V_PY" ]; then
  grep -q "^## \[$V_PY\]" CHANGELOG.md && ok "CHANGELOG.md 有 [$V_PY] 条目" || warn "CHANGELOG.md 里没有 ## [$V_PY] 条目"
fi

# ── 4. ops/*.json 语法 ───────────────────────────────────────────────────────
step "4. ops/*.json 语法(这两个文件是线上直接消费的,坏了就是全网故障)"
for f in ops/*.json; do
  if python3 -c 'import json,sys;json.load(open(sys.argv[1]))' "$f" 2>/dev/null; then
    ok "$f JSON 合法"
  else
    bad "$f JSON 语法错误 —— App 拉到坏 JSON 会静默降级,提醒/公告直接失效"
  fi
done
if [ -f ops/status.json ]; then
  ACTIVE="$(python3 -c 'import json;print(json.load(open("ops/status.json")).get("active"))' 2>/dev/null || echo "?")"
  if [ "$ACTIVE" = "True" ]; then warn "ops/status.json 的 active=true —— 故障公告横幅正在全网显示,确认这是有意的"
  else info "ops/status.json active=$ACTIVE(故障公告未开启)"; fi
fi

# ── 5. 编译 / 测试 / lint ────────────────────────────────────────────────────
step "5. 编译 / 测试 / lint(和 CI 同一组命令)"
if [ -n "$PY_CMD" ]; then
  # shellcheck disable=SC2046,SC2086
  if run_logged compile $PY_CMD -m py_compile __init__.py $(ls comfylink/*.py); then
    ok "py_compile 通过"
  else bad "py_compile 失败"; show_tail compile 20; fi

  if [ "$DO_TEST" = "1" ]; then
    # shellcheck disable=SC2086
    if run_logged unittest $PY_CMD -m unittest discover -s tests; then
      ok "unittest 全绿($(grep -oE 'Ran [0-9]+ tests' "$LOGDIR/unittest.log" | tail -1 || true))"
    else
      bad "unittest 有失败"
      show_tail unittest 40
      fix "注意:CI 用的是 python -m unittest,不是 pytest"
    fi
  else
    warn "已跳过 unittest"
  fi
fi

if command -v ruff >/dev/null 2>&1; then
  if run_logged ruff ruff check .; then ok "ruff check 干净"
  else bad "ruff check 有问题"; show_tail ruff 30; fi
else
  warn "没装 ruff,跳过 lint(CI 会跑,别指望蒙混过关)"; fix "pip install ruff 或 brew install ruff"
fi

if command -v npx >/dev/null 2>&1; then
  if run_logged acorn npx --yes acorn --ecma2022 --module web/comfylink.js; then ok "web/comfylink.js 语法 OK(acorn, ESM)"
  else bad "web/comfylink.js 语法错误"; show_tail acorn 20; fi
else
  warn "没有 npx,跳过 web/comfylink.js 语法检查(CI 会跑)"
fi

# ── 摘要 ────────────────────────────────────────────────────────────────────
printf '\n\033[1m═══ 即将发布 ═══\033[0m\n'
printf '  插件版本   %s\n' "${V_PY:-?}"
printf '  commit     %s\n' "$(git rev-parse --short HEAD)"
printf '  分支       %s\n' "$BRANCH"
printf '\n\033[1m结果:\033[32m%d 通过\033[0m · \033[33m%d 警告\033[0m · \033[31m%d 失败\033[0m\n' "$PASSN" "$WARNN" "$FAILN"

if [ "$FAILN" -gt 0 ]; then
  printf '\n\033[31m✗ 有硬性问题,先修完再发。\033[0m\n\n'
  exit 1
fi
printf '\n\033[32m✓ 检查通过。下一步(本脚本不会替你做):\033[0m\n\n'
printf '  1) git push  —— GitHub 上的用户 git pull / ComfyUI-Manager 更新即可拿到\n'
printf '     版本清单 ops/versions.json 和代码在同一个提交里,push 完提醒立刻生效(不用等任何审核)。\n'
printf '\n  2) \033[1m发 Comfy Registry(单独一条通道,push 不等于发布)\033[0m\n'
printf '     仓库里【没有】publish workflow。不发的话,ComfyUI-Manager 的用户收不到这一版,\n'
printf '     只有手动 git pull 的人能拿到。\n'
printf '       pip install comfy-cli && comfy node publish     # 需要 Registry 的 API key\n'
printf '     发完去 https://registry.comfy.org 确认版本号变成了 %s\n' "${V_PY:-x.y.z}"
printf '\n  3) 中继那处 defaultPluginLatest 若也改了 —— 记得单独 push 中继(= 触发 Render 部署)\n\n'
printf '  \033[2m发版前审查已通过门禁(第 0 步)。清单和它背后的道理见 ../app/docs/release-runbook.md §2。\033[0m\n\n'
