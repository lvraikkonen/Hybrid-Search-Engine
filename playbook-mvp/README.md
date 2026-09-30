# Playbook Local Search MVP

本地知识库门户：按用户权限浏览目录、阅读中英文文档，通过 **ASK AI** 搜索并跳转原文。当前采用 Elasticsearch BM25，返回文档和原文摘录；支持原文/metadata 下载及完整快照的手工更新。

自带 130 篇文档、20 个 Mock 用户。每次业务读取使用该用户对应的 native ES 身份和 DLS；角色互斥、Public 也需有效会话。Mock 登录仅允许在 development/test 环境启用。

## 运行要求

- macOS 或 Linux（同步门禁使用 POSIX 文件锁）。
- uv：使用 `.python-version` 固定的 Python 3.12.12，依赖由 `pyproject.toml` / `uv.lock` 管理。
- Docker 和 Compose v2：ES 容器需要 2 GiB 内存，JVM 堆为 1 GiB。
- 支持 DLS 的有效 Elasticsearch 许可。首次初始化可显式激活 trial；许可失效时停止受限读取，不降级权限。
- Node.js 仅用于客户端测试，无需安装 npm 依赖。

以下命令除 `cd` 外均在 `playbook-mvp/` 中执行。ES 固定为 **9.5.4 + analysis-smartcn**，仅映射 `127.0.0.1:19200`；API 默认 `127.0.0.1:18080`。

## 已有环境：启动与使用

本机已完成初始化并保留 `.local/` 和 Docker 数据卷时：

```bash
cd playbook-mvp
uv sync --locked
docker compose up -d --wait --wait-timeout 180
uv run python -B build.py
APP_ENV=development MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080
```

打开 [本地 Playbook](http://127.0.0.1:18080/)，在右上角选择用户。日常启动直接使用现有索引和账号，无需重新初始化或批准开放。

- 点击顶部 **ASK AI**（窄屏为右侧浮动入口），或按 `⌘K` / `Ctrl+K` 打开搜索弹窗。
- 输入问题，点击结果标题跳转文档；按 `Esc`、点击关闭按钮或遮罩关闭弹窗。
- 左侧目录、正文、原文和下载均受当前身份权限约束。切换用户会清空已有内容。
- 搜索默认最多返回 5 篇文档，按相关性得分降序排列，并高亮标题和摘录中的匹配词。
- 相关性得分是本次查询的 BM25 分数，仅供同次结果比较，不代表答案正确率。请打开原文确认是否包含答案；库中没有答案的问题也可能返回匹配文档。

修改 `src/` 后重新执行 `uv run python -B build.py` 并刷新页面。`web/` 是可重建的应用壳，只包含三个静态资源。修改 Python 后需重启 API，会话随之失效，重新选择用户即可。

停止 API：在运行终端按 `Ctrl+C`。需要停止 ES 时：

```bash
docker compose stop es
```

保留 `.local/` 和命名卷 `playbook-mvp_es_data`、`playbook-mvp_es_certs`，以便下次启动及同步失败恢复。

## 新环境：首次初始化

仅用于尚未导入数据、建立账号和批准开放的环境。已有环境使用上一节；后续数据变更使用下方“更新数据”流程。

### 1. 安装依赖并启动 Elasticsearch

```bash
cd playbook-mvp
uv sync --locked
uv run python -B tools/es_local.py prepare
docker compose build es
docker compose config --quiet
docker compose up -d --wait --wait-timeout 180
uv run python -B tools/es_local.py bootstrap --start-trial
uv run python -B tools/es_local.py check
```

`prepare` 创建私有凭据并保留已有密码；`bootstrap` 获取公开 CA 并初始化合成能力探针。`--start-trial` 表示明确同意在 Basic 许可下激活 trial；已有合适许可时不会重复激活。ES 使用 TLS 和 CA 校验，管理员仅用于初始化、维护。

### 2. 导入并检查默认关闭状态

```bash
uv run python -B tools/ingest_playbook.py dry-run
uv run python -B tools/ingest_playbook.py init
uv run python -B tools/ingest_playbook.py import
uv run python -B tools/ingest_playbook.py check --expect closed
```

原始 `data/playbook-sampledata.sql` 只做静态解析，不执行 SQL。导入默认不开放；ACL 无法解析时拒绝导入和开放。检查差异、隔离结果与源快照后，再执行下一步。

### 3. 显式批准实验开放并建立用户

下列 SHA-256 对应仓库自带 SQL。将 `--approved-by` 替换为实际操作者，确认此快照可用于本地实验后执行：

```bash
uv run python -B tools/ingest_playbook.py release \
  --snapshot cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821 \
  --approved-by '实际操作者'
uv run python -B tools/ingest_playbook.py check --expect open
APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py init
APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py check
uv run python -B build.py
APP_ENV=development MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080
```

实验开放使用独立 `lab_release` 记录；上游发布状态仍是 `unverified`。打开 [本地 Playbook](http://127.0.0.1:18080/) 后选择用户即可使用。

## 验证

不连接真实 ES 的回归（HTTP 测试会使用本机临时端口）：

```bash
uv run python -B build.py
uv run python -B -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/test_s5_client.mjs
```

原 SQL 快照已初始化并显式开放时，可运行完整自动验收：

```bash
bash tools/check_mvp.sh
# 可选：同时验证 ES 容器重启恢复，会短暂中断本地查询。
bash tools/check_mvp.sh --restart-es
```

完整验收包含构建、Python/客户端回归、20 身份真实 DLS、HTTP、冻结金标和隔离变更检查。变更检查写入 `playbook-s6-validation` / `playbook-s6-archive` 及 `pb_s6_*` 测试身份；业务索引和业务账号保持原快照。浏览器交互需要另行实测。

V1 使用用户批准的 32 条问题、3 条无答案样例及五类代表身份，目标为各类可回答问题 Hit@5 ≥ 90%。当前指标与运行环境见 [评测报告](reports/S6_EVALUATION.json)，完整证据和限制见 [验证台账](VALIDATION_STATUS.md)。新业务快照需要另行审核相应的权限预期与金标，不能直接沿用 V1 验收结论。

## 更新数据

取得一致的六表完整导出后，先生成受控快照和差异计划。示例使用自带 SQL；实际更新时将 `--sql` 指向新的完整导出，保留原始基线文件。

```bash
export APP_ENV=test MOCK_AUTH_ENABLED=1
uv run python -B tools/sync_playbook.py export --complete \
  --sql data/playbook-sampledata.sql --out .local/s6/review/source.json
uv run python -B tools/sync_playbook.py plan --source .local/s6/review/source.json
```

审核输出的正文、ACL、菜单、身份授权、撤回和删除差异后，将计划 ID 与操作者替换为实际值：

```bash
uv run python -B tools/sync_playbook.py apply \
  --plan '已审核的64位plan_id' --approved-by '实际操作者' --approve-release
```

`--approve-release` 显式批准本次实验开放；省略时受影响的新版本保持关闭。包含删除时，确认删除列表后另外添加 `--allow-delete`。可信身份授权与 `withdrawn_document_ids` 由操作者在受控快照中维护，浏览器不能提交权限。

同步期间 API 暂停读取；成功后旧会话失效，重新选择用户。失败时保留门禁，使用同一计划和原开放/删除决定重试，不能删除门禁文件强行恢复。`.local/` 内的凭据、计划、决定和批次是恢复资料，应随数据卷保留。

只回滚正文时，从历史批次生成回滚包，再按上述 `plan` / `apply` 流程审核：

```bash
uv run python -B tools/sync_playbook.py rollback-content \
  --from-batch '历史64位批次ID' --out .local/s6/review/rollback.json
uv run python -B tools/sync_playbook.py plan --source .local/s6/review/rollback.json
```

回滚保留最新 ACL、身份授权、撤回和删除。首次完成受控同步后，旧导入/开放及身份初始化入口会拒绝执行，防止覆盖新权限。详细规则见 [同步契约](CONTRACTS.md#9-s6-手工完整快照与同步窗口)。

## 文件与故障排查

| 路径 | 用途 |
|---|---|
| `data/`、`fixtures/`、`MANIFEST.json` | 原始 SQL、独立回归基线和来源指纹；保留兼容副本供一致性检查 |
| `api.py`、`runtime.py`、`documents.py`、`mock_auth.py` | API、同步门禁、文档转换与身份规范化 |
| `src/`、`build.py` | 页面源码与构建器 |
| `compose.yaml`、`infra/`、`tools/` | 本地 ES、导入、身份维护、同步与验收工具 |
| `config/synonyms.json`、`evaluation/` | 已批准同义词、候选问题与冻结金标 |
| `.local/`、`web/`、`.venv/` | Git 忽略的私有状态、构建产物及 uv 环境；只有后两项可重建 |
| `reports/`、`VALIDATION_STATUS.md` | 验收证据和开发记录 |

旧完整 HTML 与离线数据仅作来源/UI 对照。在线入口为同源 API 服务，静态白名单只开放 `web/` 中的应用壳；不要把项目根目录作为静态网站提供。

- 页面无法访问：确认 API 终端仍在运行，检查端口；可用 `--port` 改为其他空闲端口。
- 页面 404：执行 `uv run python -B build.py`。
- 会话失效：API 重启或成功同步后重新选择用户；Mock 需要同时设置 `APP_ENV` 和 `MOCK_AUTH_ENABLED=1`。
- ES 或许可错误：执行 `docker compose ps --all`、`docker compose logs --tail 80 es`，并运行 `uv run python -B tools/es_local.py check` 核查 TLS/DLS。保持认证与 TLS 开启。
- 同步失败返回 503：按原计划重试；恢复过程及门禁规则见 [契约](CONTRACTS.md)。

本地 MVP 的边界：尚未接入生产认证和上游发布语义；没有实时自动同步、LLM 回答或可靠的无答案识别。真实业务独立问题集、新业务快照在主索引的切换、生产负载与多浏览器验证仍待补充。

开发约束见 [AGENTS.md](AGENTS.md)，数据与权限定义见 [CONTRACTS.md](CONTRACTS.md)，阶段范围见 [Roadmap](PLAYBOOK_ASK_AI_ROADMAP_v1.md)。
