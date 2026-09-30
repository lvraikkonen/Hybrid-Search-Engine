# Playbook MVP 验证台账

日期：2026-09-29。当前阶段：**S0–S6本地MVP已完成**。下方保留各阶段当时的记录并追加S6实测；用户批准的V1金标已冻结并达90%目标，三组同义词未变。原包reports保留原样；主业务快照未更新，变更验证使用隔离合成数据。生产接入仍未完成。

## S0：实际修改了什么

工作分支：`playbook-local-search-mvp`。范围仅 `playbook-mvp/`。

| 文件 | 实际变化 |
|---|---|
| `.python-version` | 固定本机已有的 uv CPython 3.12.12；未下载解释器或安装应用依赖 |
| `AGENTS.md` | 落实阶段边界、uv、来源保护、角色互斥、禁止管理员代查/补 ACL/全量前端回退等约束 |
| `CONTRACTS.md` | 冻结输入优先级、主路径、源实例/指纹、25/99 追溯、DocumentRecord、Principal、开放与 API 契约及未决项 |
| `tests/test_s0_baseline.py` | 新增两个标准库 unittest：SQL/派生包/260 个原文重建一致性，以及 7 个 CLI 正负场景 |
| `README.md` | 改为 S0 完成后的入口，统一 uv 命令，区分现有离线产物、历史浏览器结果与后续在线能力 |
| `PLAYBOOK_ASK_AI_ROADMAP_v1.md` | 仅勾选 S0-01 至 S0-08，更新材料现状及单阶段执行边界 |
| `VALIDATION_STATUS.md` | 新建本台账，记录实际命令、退出条件、未验证项及 S1 入口 |

原 SQL、Mock fixture、权限 oracle、原抽取/规范化脚本、旧 HTML/源码/报告均未改写。没有创建 DocumentRecord/API 实现、ES 用户、索引或实验开放记录；新契约是后续阶段的实现依据。

## S0：实际运行了哪些命令和测试

工作目录为 `/Users/claus/Hybrid-Search-Engine/playbook-mvp`，除下表注明外。Python 入口统一通过 uv，测试内部子进程继承其 `sys.executable`。

| 实际命令 | 结果 |
|---|---|
| `git status --short --branch`（仓库根） | 当前为目标分支，`playbook-mvp/` 本来就是未跟踪目录；本轮未提交 Git |
| `rg --files`、`cat`/`sed`、只读 Python 盘点脚本及 CodeGraph 查询 | 核对目录、输入/配置、来源指纹和已有规范化/ACL 调用；最初只读盘点使用系统 python3，阶段运行与测试随后统一切换到 uv |
| `uv --version` | `uv 0.11.8` |
| `uv run --help` | 确认本机 uv 的运行参数；未安装依赖 |
| `uv python list --only-installed --offline`（仓库根） | 首次受工作区外缓存权限限制；获准访问后重跑成功，确认本机已有 uv 管理的 3.12.12，无解释器下载 |
| `uv run python -B -c 'import sys; print(sys.version); print(sys.executable)'` | 3.12.12，路径为 `/Users/claus/.local/share/uv/python/cpython-3.12.12-macos-x86_64-none/bin/python3.12` |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 首次 39 项通过（2.014 秒）；补充 Portal 角色状态对照后最终复核 **39 项通过，1.936 秒，退出码 0** |
| `uv run python -B -`（只读文档与指纹核对脚本） | 5 份文档的链接/代码围栏正常；仅 S0 的 8 项任务被勾选；固定解释器一致；原 SQL、抽取/Mock 工具、UI 源码及 HTML 共 8 个文件指纹未变化 |

39 项由原有 37 项 Mock 测试和新增 2 项 S0 基线测试组成，不包括浏览器测试。测试内部实际执行：

```text
<uv 选定的 sys.executable> -B tools/build_dataset.py --sql <原 SQL 绝对路径> --out <临时目录>/mock
<uv 选定的 sys.executable> -B tools/extract_playbook_sql.py <原 SQL 绝对路径> --out <临时目录>/extract
<uv 选定的 sys.executable> -B mock_auth.py <下列 CLI 参数>
```

临时目录由 `TemporaryDirectory(prefix='playbook-s0-')` 创建并自动清理，不覆盖任何原 fixture 或预期。没有执行原始 T-SQL；独立 oracle 在内存 SQLite 中查询解码后的关系表。

| CLI 参数 | 环境 | 实际退出码与内容 |
|---|---|---|
| `list` | development，mock=1 | 0；20 个 profile ID 与 fixture 一致 |
| `whoami --profile pb_msft_01` | development，mock=1 | 0；完整规范化对象一致，唯一 MSFT 角色、有效双业务 |
| `visible --profile pb_partner_nce_01` | development，mock=1 | 0；102 个允许 ID 与 oracle 逐一一致 |
| `visible --profile pb_partner_ospa_01` | development，mock=1 | 0；104 个允许 ID 与 oracle 逐一一致 |
| `es-metadata --profile pb_partner_both_01` | development，mock=1 | 0；与候选 metadata 一致，未创建账号 |
| `whoami --profile pb_msft_01` | production，mock=1 | 2；`mock_disabled` |
| `list` | development，mock=0 | 2；`mock_disabled` |

其他实测证据：

- SQL SHA-256 与契约固定值一致；原 fixture 的包清单指纹一致，三个 `data/` 兼容副本逐字节一致。
- 从 SQL 重建的 7 个 fixture 和 2 个报告共 9 个文件，与原包输出逐字节一致；不是用 `can_read_acl()` 生成自己的预期。
- Document 130、MenuItem 164、关系 130、Role 3、授权关系 195、Notice 4；文档集合按真实 ID 对照，不假定连续。
- 260 个重新导出的正文源字段通过 SQL 解码保真检查，并与旧 Portal 原文逐字符/换行及 SHA-256 对照一致。
- 130 篇核心 ACL 和角色状态在抽取结果、Mock/Portal 来源间一致；25/99 的菜单、业务、角色及关系 ID 单独断言。
- 2,600 个权限组合、非法混合/重复角色、内部角色无菜单豁免、Public 跨业务及身份失效门槛通过原有测试。

## S0：哪些退出条件已经满足

| 条目 | 结论与证据 |
|---|---|
| S0-01 独立边界 | 本目录 AGENTS、uv 入口及源码范围明确；其他演示未改造 |
| S0-02 来源一致与旧基线退役 | 固定 SQL 指纹，重建与原包一致；契约记录旧 Excel 42/88、146 映射和 v1 混合角色不可沿用 |
| S0-03 身份/ACL 规则 | 互斥角色、内部默认双业务且不豁免菜单、Partner 显式业务、Public 例外已写入契约和约束 |
| S0-04 最小契约 | DocumentRecord、Principal、SearchRequest/SearchResponse、读取路由及错误边界已冻结；未把文档契约冒充已实现接口 |
| S0-05 发布与开放分离 | unverified 原值与默认 false 的 lab_release 独立；批准绑定快照且须显式操作，本阶段没有开放 |
| S0-06 首版能力 | 固定为文档级 BM25 结果、原文摘录及打开正文，范围排除生成与多轮上下文 |
| S0-07 未决项与安全处理 | 发布、Active/Role.Status、多菜单、继承、时区、格式及语言事实明确区分；遇到歧义隔离或拒绝 |
| S0-08 文档与台账 | AGENTS、CONTRACTS、本台账、README 和 Roadmap 已关联 |

阶段退出条件满足：Document 25/99 可追溯到正式菜单、关系和三个角色授权；上游事实、已确认业务规则、下游派生与实验假设已分开；旧文件的退役范围已明确。

## S0：哪些仍未验证

- ES 镜像/客户端/分析插件版本兼容、Docker 运行与资源、TLS/认证、许可证/到期行为、真实 DLS 能力及重启持久化。
- DocumentRecord 实际转换、ES mapping/幂等导入、检索文本尤其复杂表格的语义保留、显式实验开放操作。
- 20 个真实 ES 身份、按 ID/批量/搜索读取与写入拒绝，以及会话/API 并发隔离。
- 在线前端载荷移除、跨标签页/身份切换竞态、浏览器下载及真实本机渲染。未运行旧页面构建或安装其依赖。
- 真实业务问题集、经审核 Golden Set、相关性指标与性能；当前不宣称 Hit@5≥90%。
- 正文/ACL/身份撤权、删除、批次切换、回滚与同步窗口；生产身份协议、发布枚举和正式时区。

这些属于 S1–S6 或后续生产集成，不作为 S0 的已通过项。原包的 `reports/TEST_RESULTS.txt`、浏览器报告及 screenshots 保持历史身份。

## S0 完成时的下一阶段入口：S1（现已执行）

读取本台账、AGENTS、CONTRACTS 和 Roadmap 的 S1-01 至 S1-07；下一次只执行 S1：

1. 核对实际 Mac 架构、Docker 可用资源、磁盘及回环端口；不能把 uv 二进制的架构当成主机架构证明。
2. 选择实际可拉取、与客户端及中文分析方案兼容的固定 ES 镜像；本阶段未沿用旧 Starter 版本。
3. 创建并实际验证单节点、持久卷、认证、TLS/CA 与健康检查；管理员、导入和查询身份分离。
4. 检查许可证和 DLS 实际能力；如需 trial，执行显式操作并记录到期日。无法支持时记录阻塞并单独决策，不静默改用管理员或关闭 DLS。
5. 验证无凭据拒绝、证书校验、受限读取能力及重启持久性，记录实际命令与证据。

S1 不顺带执行 S2 的全量入库、S3 的 20 用户验收、S4 API 或 S5 页面改造。

## S1：实际修改了什么

工作分支仍为 `playbook-local-search-mvp`，修改范围仅 `playbook-mvp/`。本次检查完成于 2026-09-29 07:28:39 UTC，机器可读证据为 [reports/S1_CHECKS.json](reports/S1_CHECKS.json)。

| 文件 | 实际变化 |
|---|---|
| `.gitignore` | 忽略 `.local/`，其中保存本机密码、公开 CA 副本和初始化状态 |
| `infra/elasticsearch/Dockerfile` | 固定 ES 9.5.4 tag+摘要，在构建阶段安装匹配的 `analysis-smartcn` |
| `compose.yaml` | 新建独立单节点 ES、证书初始化服务、两个命名卷、HTTP/transport TLS、密码文件、CA 健康检查及回环 19200 端口 |
| `tools/es_local.py` | 标准库实现 prepare、显式 trial/bootstrap、真实能力检查；不引入应用依赖或在线 API |
| `reports/S1_CHECKS.json` | 保存本次真实许可、插件、token、身份集合、拒绝结果、TLS 和重启证据，不含密码 |
| `README.md` | 更新 S1 状态、首次启动/日常启停/重建说明、身份范围、许可到期和未验证项 |
| `PLAYBOOK_ASK_AI_ROADMAP_v1.md` | 仅新增勾选 S1-01 至 S1-07，S2–S6 仍未勾选 |
| `VALIDATION_STATUS.md` | 追加本阶段的五类记录，保留 S0 的历史证据 |

实际运行状态：`playbook-mvp-es-1` healthy，监听 `127.0.0.1:19200`；`playbook-mvp-certs-1` 成功退出。另一个名为 `elasticsearch` 的原有容器仍监听 9200，未修改它的配置、索引或卷。

本阶段只写入 `playbook-s1-capability` 的 14 条合成数据，创建 `pb_s1_importer`、`pb_s1_nce`、`pb_s1_ospa` 三个 native 用户及 importer/reader 两个安全角色。它们不是业务 profile，不计入 20 用户。合成数据的开放批准只用于能力检查；原 SQL、fixtures、oracle、旧 UI/HTML/报告、S0 契约及真实文档开放状态均未改变。

## S1：实际运行了哪些命令和测试

除特别注明外，工作目录为 `/Users/claus/Hybrid-Search-Engine/playbook-mvp`。

| 实际命令/检查 | 结果 |
|---|---|
| `git status --short --branch`、CodeGraph 查询、`cat`/`sed` | 核对阶段边界和现有实现；索引未覆盖 S1 代码，无可复用 Docker/TLS 实现。本目录原先未跟踪，本轮未提交 Git |
| `uname -m`、`sw_vers`、`command -v docker`、`docker version`、`docker compose version` | x86_64；macOS 15.7.5 (24G624)；Docker 29.8.1 / Desktop 4.93.0；Compose 5.5.1 |
| `docker info --format '{{json .}}'` 及后续定向格式、`docker ps --format ...`、`docker image ls --format ...`、`docker inspect elasticsearch --format ...` | Docker VM 8 CPU、8,322,748,416 字节内存（约 7.75 GiB）；原 ES 已占用 9200、限制 2 GiB |
| `df -h .`（仓库根）、`lsof -nP -iTCP:19200 -sTCP:LISTEN`、uv 只读磁盘盘点 | 宿主卷可用 320 GiB；19200 无监听（lsof 退出 1）；Docker 磁盘位置已确定 |
| `docker image inspect docker.elastic.co/elasticsearch/elasticsearch:9.5.4 --format ...`、`docker manifest inspect docker.elastic.co/elasticsearch/elasticsearch:9.5.4` | 本地 amd64 与官方固定 tag 可用，官方 manifest 同时列出 amd64/arm64；没有执行 arm64 验证 |
| `docker pull docker.elastic.co/elasticsearch/elasticsearch:9.5.4@sha256:82ac14f43fe701992e601f4cc81e1c0d7dbc5a2576d8cd736006452925df4026` | 成功；固定摘要镜像已在本机，无需重复下载全部层 |
| `uv run python -B tools/es_local.py prepare`（两次） | 创建 4 个独立随机密码；第二次保留值并将全部文件收紧为 600、目录 700，无密码输出 |
| `docker compose config --quiet`（三次） | 静态配置解析通过；不将此结果当作运行验收 |
| `docker compose build es` | 成功安装 `analysis-smartcn`，生成 `playbook-es:9.5.4-smartcn` |
| `docker compose up -d --wait --wait-timeout 180`（三次） | 前两次暴露初始化问题，修正后第三次 healthy，详见下文 |
| `docker compose logs --tail ... es/certs`、`docker inspect playbook-mvp-es-1 --format '{{json .State.Health}}'`、`docker compose ps --all` | 定位并验证启动修复；只读取本阶段新容器日志 |
| `uv run python -B tools/es_local.py bootstrap` | 预期退出 1：实际为 Basic，未带 flag 时拒绝自动激活 trial，没有创建合成账号/索引 |
| `uv run python -B tools/es_local.py bootstrap --start-trial` | 退出 0；明确激活 trial，实际 API 返回 acknowledged/trial_was_started=true；建立合成能力环境 |
| `docker compose exec -T es df -h /usr/share/elasticsearch/data` | Docker VM 数据文件系统 458G、使用 4.2G、可用 430G |
| `docker compose exec -T es stat -c '%a %U:%G %n' ...` | 卷内启动密码和服务私钥为 root:root 640；CA 私钥 root:root 600 |
| `docker compose exec -T es bin/elasticsearch-plugin list` | 输出 `analysis-smartcn`；API 另验证插件和 ES 均为 9.5.4 |
| `uv run python -B tools/es_local.py check --restart` | **退出 0，完整 S1 实测通过**；内部真正执行 `docker compose restart es` 并等待健康恢复 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | **39 项通过，1.945 秒**，原 2,600 个 ACL 组合及 S0 来源基线仍通过 |
| `git check-ignore -v playbook-mvp/.local/elastic_password playbook-mvp/.local/pb_s1_nce_password`（仓库根） | 本地凭据由新 `.gitignore` 忽略 |
| `uv run python -B -`（只读凭据/权限/SQL 指纹核对） | 4 个密码未出现在当时 50 个项目文件中，目录/文件模式正确，SQL SHA-256 未变 |
| `uv run python -B -`（最终文档/报告核对）、`git diff --stat`、`git status --short --branch` | 5 份文档的本地链接/代码围栏正常；Roadmap 恰好勾选 S0+S1 共 15 项；真实报告 passed 且重启前后 reader 结果一致；50 个项目文件无凭据。原有已跟踪文件无 diff，本目录仍未跟踪 |

Docker socket 和 uv 工作区外缓存首次被沙箱拒绝，获准访问后重跑。早期只读环境盘点曾使用系统 python3；所有阶段脚本及测试通过 uv 的 CPython 3.12.12 执行，未安装 Python 第三方包。

首次启动失败的原因和修复：ES 9.5.4 拒绝 644 的 `ELASTIC_PASSWORD_FILE`。最终将宿主机密码全部设为 600，通过 root 初始化服务复制到命名卷，ES 读取 root 组 640 副本。第二次初始化发现基础镜像没有 `cmp`，改用 Bash 内建比较且不输出秘密；第三次启动通过。未关闭任何认证、TLS 或权限检查，未删除数据卷。

环境和版本记录：

| 项目 | 实测值 |
|---|---|
| 基础镜像 | `docker.elastic.co/elasticsearch/elasticsearch:9.5.4@sha256:82ac14f43fe701992e601f4cc81e1c0d7dbc5a2576d8cd736006452925df4026` |
| 实际构建镜像 ID | `sha256:cb466f002dae532ee2960591bf4dfdf517c7c45c397e163c49319b2dbf1d2233` |
| ES / Lucene / 插件 | 9.5.4 / 10.5.1 / analysis-smartcn 9.5.4 |
| 当前客户端 | uv CPython 3.12.12 的 urllib + ssl，真实 REST 请求通过；没有 ES Python 客户端依赖 |
| 资源 | 单节点 2 GiB 容器上限、1 GiB heap，HTTP 仅回环 19200，无 transport 宿主机端口映射 |
| 数据/证书卷 | `playbook-mvp_es_data` / `playbook-mvp_es_certs`；Docker VM 内 `/var/lib/docker/volumes/<卷名>/_data` |
| Docker 磁盘位置 | `/Users/claus/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw`，不直接编辑该虚拟磁盘 |
| 许可 | 从 active Basic 显式变为 active trial；激活 2026-09-29 07:27:15.030 UTC |
| **Trial 到期** | **2026-10-29 07:27:15.030 UTC**（洛杉矶当地 00:27:15，PDT） |
| TLS | 正向协商 TLS 1.3；服务证书到期 2027-09-29 07:24:16 UTC；指纹见 JSON 报告 |

实际 token 记录：

| 输入 | `smartcn` 输出 |
|---|---|
| 合同的生成 | `合同`、`的`、`生成` |
| Azure 预留实例 | `azur`、`预留`、`实例` |
| NCE OSPA 订阅取消 | `nce`、`ospa`、`订阅`、`取消` |

固定 Smart Chinese 为当前中文基线，重建方式是构建 Dockerfile 后重新启动和检查，不在运行容器里手工装插件。英文会词干化（如 Azure→azur）；S2 的同义词与 S4 查询处理必须考虑这个实际行为，尚未据此宣称相关性达标。

## S1：哪些退出条件已经满足

| 条目 | 结论与实际证据 |
|---|---|
| S1-01 环境条件 | 架构、Docker 资源、磁盘位置/剩余空间、端口冲突已核对；独立使用 19200 |
| S1-02 可复现版本 | 官方固定 tag/摘要实际拉取成功；ES/插件 9.5.4 与当前标准库客户端真实请求兼容 |
| S1-03 安全单节点 | 命名卷、认证、HTTP/transport TLS 配置已运行；健康检查通过，外部端口绑定实际检查为 127.0.0.1 |
| S1-04 身份分离 | importer/native reader 的 `_authenticate` 与唯一角色核对通过；管理员只做安全/许可/维护 API，不发送文档搜索；没有运行时 API 或管理员回退路径 |
| S1-05 真实 DLS | active trial 下两个 reader 的 search/get/mget 结果逐 ID 正确；无应用 ACL 过滤遮掩错误；记录许可到期 |
| S1-06 中文方案 | 实际安装和插件版本检查通过，三个术语 `_analyze` 结果已记录 |
| S1-07 重启持久化 | 真正重启 ES，前后 cluster/index UUID、全文内容指纹、native 用户/metadata、完整角色定义和 TLS 证书一致；重启后再次通过 DLS 检查 |

具体能力检查：

- 正确 CA 对 localhost 和 127.0.0.1 均通过；系统默认不信任该 CA 时拒绝，错误主机名同样因证书校验失败被拒绝，未使用 verify=false 或 `-k`。
- 无凭据搜索实际返回 401。NCE 和 OSPA reader 各只得到对应受控文档加两个跨业务 public 文档，精确 ID 集合分别为 `[nce, public_nce, public_ospa]`、`[ospa, public_nce, public_ospa]`。
- 14 个探针涵盖角色不符/缺失、角色无效、业务不符、未开放、ACL unresolved、菜单 inactive、批准快照过期、源快照变化、批准人/时间缺失；public 不能跳过前置门槛。
- 重启前后各执行两身份×14 个逐 ID GET 和两个 14-ID 的 `_mget`，允许结果 200/found=true，拒绝结果 404/found=false；各执行一次无附加 ACL 的 match_all，逐 ID 比对预期。
- 两个 reader 写文档、修改 reader 角色、将自己提升为 importer 均返回 403；importer 修改安全角色或创建其他索引也返回 403。
- 导入身份成功建立 mapping、写入/refresh 并读回全部 14 个探针；集群 green，重启后正文指纹一致。

S1 的四项硬退出条件——TLS 验证、匿名拒绝、真实读写、DLS 能力——均满足。S0 原有数据回归仍通过。没有把这次合成测试当成 S3 的 20×130 业务权限验收。

## S1：哪些仍未验证

- 许可自然到期后的 ES 行为、续期或证书轮换；当前只有 active trial 实测。到期后须先解决许可，不能自动申请新试用或放宽 DLS。未来 API 的许可故障 503 属于 S4。
- Apple Silicon、生产部署、多节点、备份恢复、升级迁移；Docker `down` 后重建、整台 Docker Desktop/宿主机重启和删除卷恢复未测试。本轮验证的是该 ES 容器的真实 restart。
- 真实 130 篇 DocumentRecord 的 mapping/转换、260 个原文的 ES 往返、幂等导入、隔离清单、显式开放及真实角色状态投影，属于 S2。
- 20 个业务 native user、2,600 个 ES 逐文档权限期望、全部角色专属反例和撤权/身份变更，属于 S3/S6。S1 的固定合成快照 DLS 不能直接替代正式策略。
- 在线会话/API、并发身份隔离、前端按需读取、跨标签页/迟到响应、浏览器下载及本机页面构建，属于 S4/S5，仍未执行。
- 真实业务问题集、经业务审核的 Golden Set、同义词审核、相关性和性能。当前没有金标，未验证 Hit@5 或其他质量阈值。

## S1 完成时的下一阶段入口：S2（现已执行）

只在用户授权后推进 S2；先读取 AGENTS、CONTRACTS、本文和 Roadmap S2-01 至 S2-10，并复查 ES/trial 可用。

1. 复用既有 SQL 解码与原文保真检查，形成 130 条 DocumentRecord 和正式 ACL 来源证据；按契约隔离未知/歧义状态。
2. 在独立于 `playbook-s1-capability` 的正式索引中定义精确 mapping、检索文本与 ACL 投影；为 S2 导入身份明确授权该索引，不能扩大 reader 或使用管理员代查。
3. 实现 dry-run、错误/隔离报告、可重复导入与中英原文往返校验；验证同源重复导入仍为 130 篇、25/99/148 正确、复杂表格有语义证据。
4. 以独立显式操作记录真实快照的实验开放，默认仍为未开放；为 S3 完整 DLS 留下可验证的开放/关闭记录。
5. 基于实测 token 整理待业务审核的术语/同义词和候选问题，不把开发候选或 ACL oracle 宣称为 Golden Set。

S2 不自动进入 S3 的 20 用户全量验收、S4 API 或 S5 页面改造。当前容器保持运行，未清理或销毁卷。

## S2：实际修改了什么

工作分支 `playbook-local-search-mvp`，范围仍仅 `playbook-mvp/`；本阶段最终 ES 检查时间为 **2026-09-29 08:00:37 UTC**。索引 `playbook-documents-v1` 共130条，已显式批准用于本地实验，`publication_state` 全部保持 `unverified`。

| 文件 | 实际变化 |
|---|---|
| `pyproject.toml`、`uv.lock` | uv 管理 CPython 3.12.12 项目环境，锁定原有正文解析依赖 markdown-it-py 4.2.0、beautifulsoup4 4.14.3、bleach 4.1.0；未新增 API 或 ES SDK |
| `documents.py` | 复用原解码器/ACL/指纹与 `build.render()`，形成完整 DocumentRecord、合并表格检索文本、严格 mapping、active_role_ids 来源投影与隔离诊断 |
| `tools/ingest_playbook.py` | dry-run、S2 初始化、稳定 ID 幂等 Bulk、分项错误、显式开放审计和真实 ES 检查；导入不会隐式开放 |
| `tools/es_local.py` | 最小扩展：允许两个 S2 基础设施身份、NDJSON Bulk 传输、DLS 传入真实快照/批准条件；保留 S1 默认行为和逐请求身份绑定 |
| `config/synonyms.json` | 用户明确确认三组只用于本地实验，版本 s2-v1；记录操作者、时间、原回复和9个分析用例 |
| `evaluation/candidate_questions.json` | 32条文档衍生问题（28条中文/中英混排、4条英文），含源快照、场景、参考 ID；仍为待业务审核的开发候选 |
| `tests/test_s2_ingestion.py` | 7项 S2 unittest，覆盖真实正文/ACL、表格、清理、歧义拒绝、幂等/开放、分项失败、未审核同义词及错误快照 |
| `reports/S2_DRY_RUN.json`、`S2_CONTENT_SAMPLES.md` | 导入清单、指纹/隔离信息，25/63/99/148 的原文与派生检索文本抽样 |
| `reports/S2_IMPORTS.json`、`S2_RELEASES.json`、`S2_CHECKS.json`、`S2_FAILURES.json` | 追加保留实际导入、开放、ES 验证和负向失败记录；不含凭据 |
| `README.md`、`CONTRACTS.md`、Roadmap、本台账 | 更新运行入口、物理字段/分析器/开放语义、S2 勾选与 S3 入口 |

本机忽略目录 `.local/s2/<源快照>/` 保存260份不可覆盖的 UTF-8 原文导出、派生 `records.jsonl`、`mapping.json`、`quarantine.json`。`.local/s2/releases.jsonl` 保存开放意图和结果。新增 `pb_s2_importer` 与 `pb_s2_probe` 两个 native 用户，分别只有索引维护角色与 DLS reader；probe 是实验开放检查身份，不是新增业务 profile。未创建20个业务 ES 用户。

原 SQL、抽取脚本、Mock 规范化器、fixtures/oracle、旧构建器/UI/HTML/报告均未改写。原 SQL 指纹仍为 `cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821`。未实现会话/API、未构建旧全量页面，未提交 Git、未清理 Docker 卷。

## S2：实际运行了哪些命令和测试

工作目录为 `/Users/claus/Hybrid-Search-Engine/playbook-mvp`。需要 uv 缓存和本机 ES 连接的命令通过已有授权运行；没有关闭 TLS 校验或使用管理员替代 probe 读取。

| 实际命令 | 结果 |
|---|---|
| CodeGraph 查询、`rg`/`cat`/`sed`、`git status --short --branch` | 核对已有正文清理/ACL/请求链和阶段范围；保持目标分支 |
| `uv run python -B -`（S2 入口检查） | 本机 ES green、trial active；`es_local.check_readers()` 的 S1 受限读取仍通过 |
| `uv sync` | 创建本目录环境和锁文件；解析10个项目/依赖项、安装9个第三方包 |
| `uv run python -B -`（源字段/解析盘点） | 发现148的超界 rowspan；初版保守解析隔离此篇，修正为只展开现有行并记录源警告后得到130条/隔离0 |
| `uv run python -B tools/ingest_playbook.py dry-run` | 130条、隔离0、ES写入0，原文导出和mapping生成成功 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | S2初次46项通过（7.487秒）；最终 **46项通过，7.996秒，退出码0**，包含全部39项原回归 |
| `uv run python -B tools/ingest_playbook.py init` | 成功创建S2两个基础设施用户/两个角色；业务profile创建数0 |
| `uv run python -B tools/ingest_playbook.py import`（共3次） | 首次写入130；开放前再次导入0次改写；开放后再次导入0次改写；每次索引总数130、Bulk错误0 |
| `uv run python -B tools/ingest_playbook.py check --expect closed`（共2次） | 初次结构/门槛检查通过但发现查询词干不一致；修正并加入分析链对照后再次通过，probe仍0条 |
| `uv run python -B -`（开放前分析器维护，下方给出内容） | 确认 approved 数为0后 close/settings/open，仅修正查询分析器，不删除索引或数据；随后完整比对实际mapping/analysis设置 |
| `uv run python -B tools/ingest_playbook.py release --snapshot intentionally-wrong-snapshot --approved-by codex-local:S2` | 预期退出码1：实际SQL与批准快照不符，在任何 ES 写入前拒绝，记录到 S2_FAILURES |
| `uv run python -B tools/ingest_playbook.py release --snapshot cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821 --approved-by codex-local:S2` | 130篇显式批准成功，逐项错误0；操作者 codex-local:S2，时间 07:57:58.668281 UTC |
| `uv run python -B tools/ingest_playbook.py check --expect open`（共2次） | 开放后及再次导入后均通过；最终时间08:00:37 UTC，probe102条；260原文字段和全部源/派生metadata/ACL证据一致 |
| `uv run python -B -`（最终只读审计） | 8个原文件指纹未变、260个原文文件字节一致；32候选的快照/ID/业务一致；6个密码文件600、目录700，Git可见文件未含密码；130个批准对象与开放日志逐项相同；S1两个reader回归再次通过 |
| `uv run python -B -`（文档与报告一致性） | 6份文档的本地链接/代码围栏通过；仅S0–S2共25项勾选；3个直接依赖与uv锁文件一致；3次导入、4次ES检查、1次开放和1次预期失败的报告一致 |

三次导入的 UTC 时间及结果：

| 时间 | written | unchanged | count_before → count_after |
|---|---:|---:|---|
| 07:49:34.751498 | 130 | 0 | 0 → 130 |
| 07:50:13.530439 | 0 | 130 | 130 → 130 |
| 07:59:24.913111 | 0 | 130 | 130 → 130 |

首次分词输出显示 query 的 `azure` 与索引 `azur` 不一致，原因是自定义链缺少 Smart Chinese 的 `porter_stem`。已将链改为 `smartcn_tokenizer → porter_stem → smartcn_stop → playbook_synonyms`，按官方分析链顺序先词干化再解析/应用同义词。实际维护命令如下，仅在本次首次开放前执行；已有开放记录时会拒绝：

```bash
uv run python -B - <<'PY'
import json, sys
sys.path.insert(0, 'tools')
import es_local as es
from documents import IMPORTER, INDEX, index_definition, ROOT
r = es.request(IMPORTER, 'POST', f'/{INDEX}/_count', {'query': {'term': {'lab_release.approved': True}}})
es.require(r['count'] == 0, 'Refuse analyzer maintenance after release')
definition = index_definition(json.loads((ROOT / 'config/synonyms.json').read_text()))
es.request(IMPORTER, 'POST', f'/{INDEX}/_close')
try:
    es.request(IMPORTER, 'PUT', f'/{INDEX}/_settings', {'analysis': definition['settings']['analysis']})
finally:
    es.request(IMPORTER, 'POST', f'/{INDEX}/_open')
print('Updated search analyzer before release; index reopened; no data deletion')
PY
```

最终3个非同义词对照中，索引/查询链均一致：`Azure Microsoft running` → `azur/microsoft/run`、`合同的生成` → `合同/的/生成`、`NCE OSPA 订阅取消` → `nce/ospa/订阅/取消`。9个术语用例保留 token position/positionLength，三组两侧展开的 token 集合一致。依据：[Smart Chinese分析链](https://www.elastic.co/docs/reference/elasticsearch/plugins/_reimplementing_and_extending_the_analyzers)、[synonym_graph过滤器顺序](https://www.elastic.co/docs/reference/text-analysis/analysis-synonym-graph-tokenfilter)。

本快照10篇具有完全相同的中英源字段，仍全部保留。用148的 `OSPA` 做实际评分检查：zh=2.91531、en=2.8687863、both=2.91531。`dis_max(tie_breaker=0)` 取最高语言分数，不把两份相同内容相加；这验证存储/查询策略，不代表 S4 API 或搜索质量已完成。依据：[dis_max原生语义](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-dis-max-query)。

初版解析发现的148表格缺陷仍原样存在于原文及阅读HTML中：末组声明 rowspan=8，实际只有7行；仅展开存在的7行、记录每种语言两条对应单元格警告，不补第8行。63/148的目录标签、列标题、行号与单元格值保留关联；空白表示原文空白，不推导为拒绝。表格正文不参与 ACL 推导。

## S2：哪些退出条件已经满足

| 条目 | 实测结论与证据 |
|---|---|
| S2-01/02 原文与四类数据 | 130记录、260原文字段在SQL→文件→ES往返一致；全文、解析文本、业务metadata与正式ACL分别保留；原文/HTML不索引、上游状态不成为可检索正文 |
| S2-03 复杂正文 | 复用Markdown/HTML允许列表；测试保留有序步骤、63/148合并表格列/行关系；148源缺陷显式警告，抽样可复核 |
| S2-04 双语/安全HTML | null/空串、换行和混排保留；不翻译/回填/删原文；清理脚本/CSS/SVG/img事件；dis_max实测不累加语言分数 |
| S2-05/06 单文档/mapping | 稳定_id，双语同记录，130篇无额外菜单文档；严格mapping和实际分析设置一致，原始无时区datetime保持字符串、版本可追溯 |
| S2-07 ACL来源 | 130核心ACL与独立fixture一致；25→31/NCE，99→114/OSPA，148→174/OSPA；多菜单/未知角色状态等合成输入变为unresolved且active_role_ids为空 |
| S2-08 dry-run/幂等/错误 | dry-run零ES写入；三次导入总数130；相同导入保留开放；HTTP200含409分项错误的单元检查正确报错，实际三批均无错误 |
| S2-09 显式开放 | 错误快照拒绝；准确快照130篇显式开放；probe开放前0、后102；3/25/63/81为200，99/148仍404；匿名401，reader写入403 |
| S2-10 审核同义词 | 用户确认三组本地实验规则，批准记录/版本化配置齐全；9个_analyze用例和3个分析链对照通过 |

开放记录：`reports/S2_RELEASES.json`。NCE probe 的 `/_authenticate` 确认为 native user `pb_s2_probe`、唯一角色 `pb_s2_reader` 及固定 `[2]`/`[0]` metadata，未叠加应用 ACL。导入账号只用于源/索引一致性和分析/评分维护检查，未作为任何业务身份返回检索结果。

## S2：哪些仍未验证

- S3 的20个对应 native 用户、完整逐 ID 集合、20×130搜索/GET/_mget矩阵、全部角色专属/提权反例；本阶段probe数量及抽样不替代此门槛。
- S4会话、API、字段权重/Top K/摘录及API里的dis_max查询，S5按需读取与浏览器用户切换/竞态隔离；旧全量HTML保持离线参考身份。
- 真实业务问题集与经审核Golden Set仍缺失；32条文档衍生候选待审核，未冻结profile、分级相关性、无答案预期或阈值，未测Hit@5/延迟。
- 内容/ACL/身份变更、撤权、删除、快照切换、撤销批准、回滚及故障恢复仍属S6。仅离线验证变更记录恢复未开放；未对真实130篇做破坏性变更。
- Bulk分项失败已用真实API响应形状的单元检查覆盖，未在线制造部分成功、网络中断或竞争故障。开放不是整批事务；大于1,000篇/10MiB的输入拒绝，尚无分页/分块导入。
- 原始发布枚举、Active/Role.Status正式业务含义、上游时间区域、多菜单和继承仍未确认；继续采用保守实验规则。148的源rowspan缺陷已保留/标注，未修写源数据。
- 许可自然到期、证书轮换、其他CPU架构及生产部署仍未验证。S1容器重启证据保留；S2本轮未再次重启容器。trial到期仍为2026-10-29 07:27:15 UTC。

## S2 完成时的下一阶段入口：S3（现已执行）

用户授权后仅推进 Roadmap S3-01 至 S3-09。先复查trial/TLS及 `tools/ingest_playbook.py check --expect open`，使用当前索引和真实快照开放记录：

1. 从唯一 `fixtures/mock_users.json` 经既有规范化器生成20个对应 native 身份与可信metadata；20个业务身份都只有统一受限reader，单角色约束保持。
2. 使用S2的实际快照、开放条件及正式active_role_ids投影建立完整DLS；管理员只初始化，每个实际读取使用该profile自己的ES凭据并验证_authenticate。
3. 在不附加重复应用ACL的条件下，对完整可读ID集合逐项比对独立 `expected_access.json`，执行20×130搜索/GET/_mget与拒绝写入/提权检查。
4. 角色专属、非法身份及缺失授权反例放在独立合成测试中，不改真实SQL或oracle、不制造真实角色差异。
5. 保留S2导入/开放报告；S3通过后再记录S4入口，不顺带实现API或前端。候选问题审核可以讨论，但不能凭S3权限PASS宣称相关性达标。

当前ES容器保持运行，未销毁卷。S2结束，尚未进入S3。

## S3：实际修改了什么

分支仍为 `playbook-local-search-mvp`，范围仅 `playbook-mvp/`。本阶段真实ES矩阵最终通过时间：**2026-09-29 08:23:16 UTC**。未进入S4。

| 文件 | 实际变化 |
|---|---|
| `tools/es_permissions.py` | 新增S3 `init/check`：复用Mock规范化、DLS模板、HTTPS与源校验；部署20个独立native用户，执行真实/合成逐ID矩阵与权限拒绝检查 |
| `tools/es_local.py` | 允许S3合成维护身份；HTTPS请求新增显式 `password_value` 参数，每次独立构造Authorization，基础设施默认调用兼容，无业务管理员回退 |
| `tests/test_s3_permissions.py` | 5项离线边界测试：20身份metadata对照、非法角色/失效身份、环境/未知profile、私有凭据与拒绝回退、拒绝响应泄漏/缺失源数据 |
| `reports/S3_INIT.json` | 记录实际部署的20用户、统一reader DLS配置及22条独立合成文档，不含密码 |
| `reports/S3_CHECKS.json` | 完整身份/权限配置、逐ID允许集合、三条路径的验证计数、TLS/许可及不可变数据指纹；最新检查失败也会写明确failed状态 |
| `reports/S3_FAILURES.json` | 保留实际production负向拒绝及第一轮测试请求格式错误，不删除失败历史来呈现PASS |
| `reports/S2_CHECKS.json` | S3入口执行既有S2检查，追加一次open通过；S2原有四次历史记录保留 |
| `evaluation/candidate_questions.json` | 记录用户对32条问题的审核，版本v1、状态approved_validation_questions；问题、场景、参考ID与源快照不改，仅更新审核信息 |
| README、CONTRACTS、Roadmap、本台账 | 更新20身份实际映射、命令/证据、V1问题状态及S4入口；仅新增勾选S3的9项 |

新增业务用户恰好20个，native用户名与profile_id一一相同，均只有 `pb_document_reader`。业务密码在 `.local/s3/`，目录700、文件600。新增 `pb_s3_importer` 是合成索引维护身份，不是第21个业务profile；管理员只负责授权初始化和安全配置审计，不代替任何用户查询文档。

统一reader包含两条精确索引权限，每条都携带完整DLS：真实 `playbook-documents-v1` 使用S2真实快照；`playbook-s3-policy-checks` 使用独立的 `s3-synthetic` 来源和指纹 `1ac539519ceb5afa1d4eaf1308510e88648409c13ebbe98e612ce1b5a1e4cacc`。合成索引保留用于回归；S4业务查询必须固定真实索引。没有通配索引、cluster、run_as或写权限。

原SQL、身份fixture、权限oracle、原抽取/规范化脚本、旧HTML/UI和原包报告均未改写。真实130篇正文/ACL/metadata及S2批准对象前后相同；新合成ACL只存在于独立测试索引。当前三组同义词配置指纹未变；未增加同义词。uv依赖和锁文件未新增包。未提交Git、未重启或销毁容器/卷。

32条问题本轮已获用户审核，原S2“待审核”记录是历史事实。当前作为V1验证输入；尚未冻结profile、分级相关性、无答案样例或验收阈值，因此未声明相关性Golden Set或搜索质量验收完成。

## S3：实际运行了哪些命令和测试

工作目录 `/Users/claus/Hybrid-Search-Engine/playbook-mvp`；仓库状态检查在其根目录。Python全程通过uv，使用既有3.12.12环境；ES请求保持CA与主机名验证。

| 实际命令 | 结果 |
|---|---|
| CodeGraph查询、`rg`/`cat`/`sed`、`git status --short --branch` | 先追溯现有身份、DLS、HTTPS和调用方，再读取未覆盖配置；保持目标分支，原目录仍为未跟踪状态 |
| `uv run python -B tools/ingest_playbook.py check --expect open` | S3入口08:13:49 UTC通过：trial active，130记录/260原文一致，S2 probe102条 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 初次51项通过，9.197秒；最终 **51项通过，7.667秒，退出码0**，包含原46项回归 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py init` | 08:19:46 UTC完成：20业务native用户、1个统一reader、1个独立合成维护身份/角色、22条合成文档；真实文档未写入 |
| `APP_ENV=production MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py init` | 预期退出码1、`mock_disabled`；在任何凭据生成/ES操作前拒绝 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py check`（第一轮） | 返回failed：自改metadata反例漏传API必填roles，返回400而非要求的403；不将格式错误算作权限通过 |
| 同一 `check` 命令（修正后） | 08:22:56–08:23:16 UTC通过，4个有界线程；20×130真实组合、20×22合成组合均通过逐ID搜索/GET/_mget，120次拒绝操作403 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B -`（最终只读审计） | 27份唯一凭据均600、所在目录700且Git忽略，可见文件未含密码；安全API读回20用户metadata/角色与部署预期一致；实际两条reader DLS与模板完全相同；S1两reader回归和S2 probe102均通过，32问题已审核且同义词未改 |
| `uv run python -B -`（文档/指纹/计数复核）、`git diff --check`、`git status --short --branch` | 6份文档链接/围栏通过；仅S0–S3共34项勾选；8个源文件/脚本/UI指纹未变；报告7,800次真实和1,320次合成路径断言、120次拒绝计数一致；分支与修改范围不变 |

第一轮失败的根因是测试请求格式，而非取得了额外权限。修正为合法的 `roles:[pb_document_reader]` 加metadata更新后，20个身份均得到403；最后读回native用户和安全角色确认没有改变。失败记录保留于S3_FAILURES。依据：[官方用户更新API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-security-put-user)。

真实集合结果如下；每个用户均验证 `/_authenticate` 的用户名、native realm、唯一reader和完整可信metadata，同时核对 `/_security/user/_privileges` 不包含额外索引、cluster、run_as或写入权限：

| 组 | 用户数 | 每人允许 | 每人拒绝 | 单人三路径检查 |
|---|---:|---:|---:|---|
| MSFT | 4 | 130 | 0 | 各130个逐ID比较通过 |
| 21V | 4 | 130 | 0 | 各130个逐ID比较通过 |
| Partner双业务 | 4 | 130 | 0 | 各130个逐ID比较通过 |
| Partner NCE | 4 | 102 | 28 | 各130个逐ID比较通过 |
| Partner OSPA | 4 | 104 | 26 | 各130个逐ID比较通过 |

总计2,600个真实身份/文档组合（2,384允许、216拒绝），每个组合执行三条路径，共7,800次对应断言；另执行20次完整match_all集合比较。`_msearch` 每个子查询只选择单个ID，不附带ACL条件；GET和_mget验证允许文档的完整 `_source`，无权文档为found=false/404且无正文或字段载荷，未知ID采用同样外部状态。搜索响应检查超时/分片失败/子项错误，不把不完整响应当空结果。

真实记录整体指纹在检查前后均为 `e730f69e85a969f4cca1fc185aaa3bcc65b304ccf7b220553afc8b697318ecd5`，包含正文、ACL、metadata及S2开放对象。开放时间仍为 `2026-09-29T07:57:58.668281+00:00`，操作者仍为 `codex-local:S2`。用户读取只用其自身凭据；S2维护身份的全量读取仅用于前后完整性检查，不向业务请求返回结果。

合成22篇复用S1门槛样例并补充三个角色×两个业务的独立授权、空批准人与交叉角色状态反例；手写预期与生产DLS实现分开。20×22=440组合在三路径全部通过，覆盖：

- 单角色专属和业务匹配分别有效；MSFT/21V即使默认双业务，也不能读取只授予Partner的受控文档。
- Public可跨业务，但仍必须开放、ACL resolved、菜单active、源/批准快照一致、批准人非空且有批准时间。
- 无菜单授权、角色状态无效、allowed与active落在不同角色、未开放、未知ACL、失效快照、缺批准信息全部拒绝。
- 多角色/重复角色、停用/过期、未认证、未知profile、production和凭据缺失等入口情况由离线边界检查证明在ES调用前拒绝，不创建非法业务账号。

每个身份还执行6项独立拒绝操作：写真实索引正文、修改合成ACL、修改reader角色、自改为superuser、自改metadata、读取无授权的S1索引，全部403（共120项）。匿名读取401；可信CA localhost/127.0.0.1为TLSv1.3，错误CA/主机名被拒绝。trial仍于2026-10-29 07:27:15 UTC到期。依据：[DLS与单角色只读权限](https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/controlling-access-at-document-field-level)、[实际用户权限API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-security-get-user-privileges)、[逐项多查询响应](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-msearch)。

## S3：哪些退出条件已经满足

| 条目 | 结论与证据 |
|---|---|
| S3-01 身份规范化 | 恰好20个fixture身份，复用既有单角色与effective_business_categories规则；无管理员业务profile |
| S3-02/03 native与完整DLS | 20个独立用户均仅统一reader；角色/业务/active投影及开放/源快照等条件在同一DLS中，未组合宽权限角色 |
| S3-04 实际身份 | 全部_authenticate与实际权限检查通过，native用户名各不相同且metadata与规范化结果一致 |
| S3-05 完整集合 | 20次match_all逐ID比对独立expected_access，未只比较数量，未叠加应用ACL |
| S3-06 2,600组合 | 真实逐ID搜索、GET、_mget全部一致；正文及metadata一致，无权/不存在不泄漏正文 |
| S3-07 独立反例 | 22个独立合成文档覆盖角色与业务等门槛，440组合通过三路径；非法身份在离线入口拒绝，原SQL/oracle不改 |
| S3-08 拒绝写入/改权 | 120项真实ES负向操作403；读回用户/角色和真实/合成正文证明状态未改变 |
| S3-09 凭据隔离 | 20业务密码各自保存，含基础设施共27份密码唯一、权限受限、Git忽略且未出现在可跟踪文件；请求凭据独立，无管理员回退 |

S3退出门槛满足，获得M1权限化知识库基线。此结论限定CLI及ES边界，不代表已有在线会话或页面安全验收。

## S3：哪些仍未验证

- S4的同源会话、profile选择接口、API参数边界、菜单/搜索/详情/原文/下载路由及错误映射；尚未运行HTTP API或用户浏览器。4个工作线程的直接ES检查不代替API会话并发隔离。
- S5的按需正文、用户切换、取消/迟到响应、跨标签页、深链与下载安全；旧全量HTML仍仅作离线参考，没有改造或托管。
- S6的正文/ACL/身份变更、撤权、删除、批次切换、回滚和故障恢复；本轮仅验证拒绝写入，不做真实业务撤权或数据破坏测试。S3新增身份的重启持久性本轮未单独重测。
- V1的32问题已审核，但profile、相关性分级、无答案集与质量阈值仍未冻结；未执行Top K相关性或延迟评测，不宣称Hit@5达标。
- 许可自然到期/续期、证书轮换、生产SSO、生产部署及正式上游发布/状态/时区语义仍未验证；继续执行既有本地保守门槛。

## 下一阶段入口：S4

用户授权后只推进S4-01至S4-11。先复查S3完整矩阵与trial/TLS，使用当前真实索引和20个native身份：

1. 建立开发/test同源会话，仅接受预设profile_id；复用Mock规范化及S3凭据绑定，禁止客户端角色/业务授权、ES DSL、索引或凭据覆盖。
2. 实现menu/search/详情/raw/download的统一授权路径，所有读取使用当前会话用户的ES身份；仅服务专用静态目录，不公开SQL、fixtures、.local或旧全量HTML。
3. BM25搜索固定真实索引，沿用Smart Chinese与当前三组审核同义词、双语dis_max；Top K按Document去重、摘录来自原文。使用已审核V1问题做开发验证，但质量指标留待独立标注评测。
4. 会话切换/注销、错误与超时、许可不可用、并发身份及未经授权读取均建立实际API检查；ES故障不回退全量本地检索或管理员。
5. 完成后记录五类阶段结果，停在S5入口；不顺带改造页面。

S3结束，ES容器继续运行；未进入S4。

## S4：实际修改了什么

分支仍为 `playbook-local-search-mvp`，本轮范围仅 `playbook-mvp/`，按用户指令只推进S4。

| 文件 | 实际变化 |
|---|---|
| `api.py` | 新增仅本机的标准库HTTP服务：开发会话、profile标签、me/menu/BM25/search/详情/raw/download、严格输入与错误边界、响应版本及安全审计 |
| `tools/es_local.py` | 既有HTTPS传输增加只携带status的 `ESHTTPError`；原调用签名/凭据隔离/TLS校验保留，API可区分401与ES不可用 |
| `tools/check_api.py` | 新增真实HTTP联调CLI，临时端口、20个native身份、独立oracle对照、读取/并发/会话/输入/静态路径检查、32条V1问题执行与报告 |
| `tests/test_s4_api.py` | 新增6项标准库unittest，使用真实HTTP与受控ES传输故障，验证许可错误/超时、身份/DLS漂移、会话竞态、null/empty、父菜单占位等边界 |
| `reports/S4_CHECKS.json` | 最终1,127次HTTP请求的检查计数、各身份集合/合同问题结果、160次V1检索结果ID及27次语言/术语查询结果 |
| `reports/S4_FAILURES.json` | 保留首轮无结果测试词被正常分词命中的失败记录；不删去失败历史 |
| `reports/S4_INTEGRITY.json` | 8个保护文件的原指纹、真实ES记录前后一致、mapping/分析器与凭据检查、许可状态 |
| `reports/S3_CHECKS.json` | S4入口重跑S3矩阵，更新最近一次实测时间和相同结论 |
| README、CONTRACTS、Roadmap、本台账 | 记录实际API协议、启动/验证命令、完成证据、限制与S5入口；新增勾选S4共11项 |

运行时业务读取仅使用 `resolve_identity()` 返回的不可变native身份与独立凭据。每次检查用户名、唯一reader、metadata、无额外权限和两条完整DLS模板，再固定查询真实索引；不使用管理员/导入器代替用户，不从oracle或本地全量正文检索。菜单的文档集合由DLS给出，正式菜单关系只用于结构和标签可见性，不继承或补造文档授权。

搜索为文档级BM25，标题/章节/正文权重初值3/2/1，Top K默认5、最大20，双语dis_max取较高分。同义词继续使用原审核的三组，没有增删规则。DTO白名单排除内部ACL/SQL行号/凭据；详情保持契约的document外层。所有原文和Markdown下载逐字节读取已授权的ES字段，并用text/plain、nosniff、attachment和CSP控制展示。

没有新增Python依赖，没有创建生产认证、通用provider框架、缓存服务或S5页面。API没有静态文件处理器；旧HTML、SQL、fixture、`.local`与报告目录均不能通过HTTP读取。原SQL/抽取器/身份规范化器/独立oracle/旧UI均未改写。未导入、撤回、删除或重开真实文档，未更改native角色、密码、Docker容器或数据卷；未提交Git。

## S4：实际运行了哪些命令和测试

工作目录 `/Users/claus/Hybrid-Search-Engine/playbook-mvp`；所有Python运行均通过uv，复用3.12.12。S3/S4真实ES请求继续验证CA和主机名。文档中示例启动命令与实际命令的environment区别在下表明确记录。

| 实际命令 | 结果 |
|---|---|
| CodeGraph查询、`rg`/`cat`/`sed`、`git status --short --branch` | 追溯现有身份、DLS、传输、菜单与调用方，确认契约和阶段边界；分支不变 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py check` | 08:34:31–08:34:50 UTC通过；20×130真实组合、20×22合成组合三路径通过，120项拒绝，trial/TLS有效 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B -`（实际用户权限/配置只读检查） | 读回受限native用户权限结构，确认完整DLS模板、profile标签来源、菜单表和当前分析器指纹；未读管理员业务正文 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080` | 成功启动仅127.0.0.1的API；通过后述HTTP探针获得会话并按NCE身份检索 |
| `uv run python -B -`（HTTP会话/搜索探针） | `POST /api/dev/session` 和 `POST /api/search` 均200；“合同是如何生成的？” NCE首项25，未返回99；不打印cookie或ES凭据 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 初轮57项通过18.628秒；最终 **57项通过18.210秒、退出码0**，包含原51项和新增6项 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_api.py`（首轮） | 20身份/720读取及160次V1检索已通过；在无结果样例断言失败，08:48:19 UTC记录于S4_FAILURES |
| 同一 `check_api.py`（修正无结果样例后） | 08:49:07–08:49:47 UTC，1,127次真实HTTP通过 |
| 同一 `check_api.py`（最终契约复核后） | **08:50:36–08:51:15 UTC，1,127次真实HTTP通过、退出码0**；详情采用冻结的document外层并检查响应形状 |
| `uv run python -B -`（源文件/ES完整性与凭据只读审计） | 08:52:09 UTC通过；8个源文件指纹相同，130条真实记录整体指纹与S3相同，mapping/同义词未变，27份密码600/唯一/Git忽略且未出现在新代码/报告/审计日志 |
| `uv run python -B -`（报告摘要核对）、文档链接/围栏/阶段计数检查、`git diff --check` | 根据真实结果填写记录，复核新增文档和S0–S4共45项；未将S5/S6标为完成 |
| `lsof -n -iTCP:18080 -sTCP:LISTEN`、`kill -TERM 17732` | 确认并关闭本轮手工启动的临时API进程；首次沙箱禁止发信号，获准在沙箱外执行后成功；ES容器不受影响 |

首轮失败不是越权或本地回退：`s4nomatch9fbcc87a55` 经Smart Chinese拆分，数字片段实际可以匹配文档。仅将“无结果”测试词改为单个无关英文token `zzqnomatchxylophonezzq`；检索参数、权重、源数据和相关性参考没有为通过而调整。最终协议复核另发现初版详情把展示字段放在根层，已修正为CONTRACTS要求的document外层，并重跑全部回归与HTTP检查。

真实HTTP证据分类如下（它们均包含在总计1,127请求中，不与S3路径检查相加冒称同一测试）：

| 检查 | 实测结果 |
|---|---|
| 20身份会话/me/menu | 每个身份独立native核验；菜单全部document keys逐ID匹配独立oracle：内部/双业务130，NCE102，OSPA104；不是只比数量 |
| 720文档请求 | 20身份×6个ID（25/99/148/3/81/不存在999999）×6种路径（详情、双语raw、双语Markdown下载、metadata）；允许项内容/原文字节一致，禁止项与不存在项同一404响应 |
| Public跨业务 | OSPA身份读取NCE Document3、NCE身份读取OSPA Document81成功；受控25/99/148仍各自受DLS限制 |
| 同一合同问题 | NCE结果 `[25,63,24,33,42]`；OSPA `[99,101,148,92,33]`；MSFT/21V/双业务均 `[25,99,63,101,148]` |
| V1输入 | 32条×5类身份=160次检索，使用问题原language/business_scope，结果ID及摘录来源检查通过；没有新增相关性标签或计算质量达标率 |
| 语言与术语 | 9个术语/动作×zh/en/both=27次；matched_language与源字段一致，摘录逐段来自该文档S2检索文本；不把ContentEn的中文改称zh |
| 参数与来源 | 28个非法查询/路径请求、另2个非法session输入、4个Origin/Host拒绝；禁止角色/业务授权、DSL、索引、provider、凭据覆盖，拒绝重复JSON键/非法Unicode等 |
| 空结果与scope | 无关单token返回200空items；all/0/1筛选均在该身份DLS范围内，scope=1仍只返回NCE用户可见的OSPA Public内容 |
| 并发 | 4个工作线程交错两个Partner身份，40次搜索与40次禁止下载共80次；结果和session_revision不串用 |
| 会话切换/注销 | 旧cookie401，新NCE身份访问99下载404；注销后新cookie也401 |
| 匿名与文件绕过 | 6个无会话API拒绝401；10个仓库/SQL/fixture/凭据/旧HTML等路径404，未暴露磁盘正文 |
| 审计 | 1,127条日志仅有固定字段：请求/身份ID、路由标签、状态、索引/策略/转换/分析器/同义词版本、结果ID与耗时；不记录用户问题/正文/cookie/密码 |

6项新增回归使用真实loopback HTTP与替换后的ES传输，**不连接真实ES，也不修改其许可/ACL**。这些检查涵盖：

- 下游401、403许可/权限能力错误、500、连接拒绝、证书错误、传输超时、ES timed_out与分片失败；全部映射为明确401/503/504，不能当作空结果、回退本地或改用管理员。
- 错误native用户名、额外reader/superuser、错误metadata、disabled、缺失凭据、缺少DLS或索引范围漂移均停止读取；生产/禁用mock路由不可用。
- 延迟OSPA的99原文响应，在另一个真实HTTP请求中切换为NCE或注销，再释放旧下游响应；旧请求返回401且无正文。已发出的网络响应及浏览器竞态属于S5。
- 独立内存菜单反例：无权父项只显示“上级目录”，public子项保留，无权兄弟项消失；没有改真实SQL或真实ACL制造差异。
- null→204并有X-Source-Field-Null，空串→200空字节，原始脚本片段以text/plain附件返回；详情保留raw_is_null/raw_is_empty，未回填其他语言。

最终维护审计只在CLI用S2维护身份比较整体记录与mapping，不参与用户API返回。真实记录指纹仍为 `e730f69e85a969f4cca1fc185aaa3bcc65b304ccf7b220553afc8b697318ecd5`；S2实验开放对象未变。trial仍active，到期2026-10-29 07:27:15 UTC。

实现依据：[Python HTTP服务](https://docs.python.org/3.12/library/http.server.html)、[ES摘录](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/highlighting-settings)、[dis_max](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-dis-max-query)。这些文档解释机制，当前通过结论来自上面的实测。

## S4：哪些退出条件已经满足

| 条目 | 结论与证据 |
|---|---|
| S4-01 会话 | 预设20身份、标签白名单、HttpOnly随机cookie、固定过期、切换/注销、生产禁用均已实现并验证 |
| S4-02/03 请求身份 | 每次规范化/凭据/native及完整DLS检查；拒绝客户端授权/DSL/索引覆盖；并发80请求无串号 |
| S4-04 菜单 | 20身份完整集合对照oracle；真实DLS授权文档之外不增加正文，独立无权父项反例通过 |
| S4-05/06 BM25 | 默认5/最大20、文档级去重、标题/章节/双语、审核术语、dis_max0；权重标记待评测 |
| S4-07 返回数据 | 固定DTO、命中源语言、纯文本证据、来源/内容/ACL版本、request_id、retrieval_mode；不提供生成回答/分数可信度 |
| S4-08 所有读取 | 详情/raw/两种下载同一native边界，720次检查通过，非法key/静态目录不能绕过 |
| S4-09 错误 | 400/401/403/404/503/504与200空结果分别验证；异常不放权、不回退，服务/许可失效覆盖限于注入 |
| S4-10 缓存 | 无搜索/正文业务缓存，所有HTTP no-store，ES request_cache=false |
| S4-11 审计 | 实际日志记录固定ID/版本/耗时字段；凭据扫描通过，无问题全文或正文 |

S4退出门槛满足，获得M2权限内检索API。无需HTML即可完成会话→菜单→搜索→详情→原文/下载；S3与S4联合验证允许进入S5。此结论不等同于页面验收、相关性质量验收或生产认证完成。

## S4：哪些仍未验证

- S5现有页面接API、Network、用户切换清空/迟到响应、跨标签页、历史/深链、下载交互、桌面/窄屏和浏览器XSS/资源加载。旧全量HTML仍仅离线参考，本轮没有运行或托管。
- 32条问题已执行，但身份级相关性分级、无答案业务集与目标阈值未冻结；未计算或宣称Hit@5/Recall/MRR/nDCG合格，尚无S6规范的p50/p95性能报告。
- API首次只支持单进程本机开发、固定快照与1,000篇菜单上限；无持久会话、多进程、外部SSO、生产HTTP部署。新增同义词/源快照必须显式维护并复核mapping，不能把本地配置修改当作ES已更新。
- S6正文/ACL/用户授权更新、撤权、撤回、删除、回滚、批次切换与重启故障仍未执行。身份/权限核验与业务读取是顺序调用，不声称管理变更期间提供原子快照；同步窗口须在S6定义验证。
- 许可自然到期/续期、实际服务中断/网络超时和证书轮换没有制造实测；本轮使用传输层注入验证API拒绝行为。上游发布/状态/时区语义仍未核验。

## 下一阶段入口：S5

用户授权后只推进S5-01至S5-11，入口为现有 `src/template.html`、`src/style.css`、`src/app.js` 与当前 `api.py`：

1. 先复查S3和S4的CLI门槛；试验许可/TLS与当前快照仍须有效，不以本台账的历史PASS替代实测。
2. 保留现有页面布局与阅读样式，生成专用静态应用壳；移除全量portal-data、ACL oracle、内嵌全库HTML与离线下载正文。仅服务专用静态目录。
3. 用户选择接会话；菜单和文档按需读取；ASK AI用当前BM25 API显示证据卡，keyword只可筛选已授权菜单标签。无前端全库回退。
4. 切换/注销立即清空旧正文/原文/元数据/结果，取消在途请求，用session_revision与请求序号拒绝迟到结果；跨标签页共享cookie时以服务器身份为准。
5. 在浏览器运行合同25/99、Public跨业务、失败/超时、迟到响应、深链/刷新/历史/下载与窄屏检查，确认网络不再出现全量正文或凭据，再记录五类阶段结果并停止。

S4结束，不执行S5页面开发。ES容器保持原状态；临时HTTP检查服务随脚本退出，手工启动的18080服务也已关闭。需要使用API时按README的uv命令重新启动。

## S5：实际修改了什么

本轮仅推进S5-01至S5-11。分支仍为`playbook-local-search-mvp`，所有改动限定在`playbook-mvp/`；未提交Git，未开始S6。原目录本就未被Git跟踪，本轮没有把整个目录暂存或提交。

| 文件 | 实际改动 |
|---|---|
| `src/template.html`、`src/app.js`、`src/style.css` | 沿用配色、左目录、右阅读区及安全HTML样式；去掉全量portal-data、JS权限判定、全库全文检索、浏览器ACL自检与离线正文下载。新增ASK AI面板、API会话/菜单/检索/详情/raw/download、来源版本、错误重试及会话清理 |
| `build.py`、`web/index.html`、`web/style.css`、`web/app.js` | 构建改为复制三个应用资源；不再读SQL或重写旧全量HTML/portal_data。S2使用的`render()`函数源码指纹保持不变 |
| `api.py` | 固定静态白名单，仅GET/HEAD访问专用web目录的三个文件；路径/符号链接拒绝、应用CSP、no-store。菜单增加business_category/is_public展示属性；无权祖先二者均null，文档授权仍由原DLS产生 |
| `tools/check_api.py`、`tests/test_s4_api.py` | 更新新增菜单字段的DTO断言；原“首页404”测试改查`/src/template.html`，不撤掉文件绕过检查；增加无权父项新字段不得泄露的断言 |
| `tools/ingest_playbook.py` | 将依赖构建器副作用的`import es_local`改为显式`from tools import es_local`，兼容独立导入测试；未修改入库/授权行为 |
| `tests/test_s5_static.py`、`tests/test_s5_client.mjs` | 新增真实本机HTTP静态边界检查，以及故意忽略AbortSignal的客户端竞态检查；使用现有Python依赖和Node内置test/vm，无新依赖 |
| `README.md`、`CONTRACTS.md`、Roadmap、本台账、`reports/S5_*` | 更新在线启动/构建、接口展示字段、浏览器状态契约、S5证据与S6入口；保留此前阶段历史记录 |

浏览器只持有当前身份的导航、最新一次查询结果和正在阅读的一篇文档。不用localStorage存储身份或正文，也不预取下一篇。原文/下载重新调用授权接口；单篇下载经会话版本校验后生成短期Blob URL，切换立即撤销。前端没有ES凭据、DSL、管理员查询或权限oracle。

用户切换/注销立即清空正文、原文、元数据弹窗、问题、结果和目录，取消在途请求；本地身份代次、每类请求票据、视图序号及服务端session_revision共同拒绝迟到响应。收到旧请求401也不会注销新身份。跨标签页仅广播变更信号，随后读取`/api/me`；不把广播的profile当作权限。焦点/可见性恢复会核对服务端会话，pagehide清空，深链与历史重新读取。

没有改写SQL、原始抽取器、Mock规范化器、fixture/oracle、三组同义词、32条V1问题或旧离线完整HTML。未导入/开放/撤回/删除文档，未改ES用户/角色/密码/许可或容器。只读维护核对表明130条真实ES记录指纹与S4相同，mapping/分析器未变；27份凭据仍为600且受Git忽略，未出现在应用资源或新报告/审计日志中。

## S5：实际运行了哪些命令和测试

工作目录`/Users/claus/Hybrid-Search-Engine/playbook-mvp`，Python均由uv选择3.12.12。检查初始门槛使用真实ES与TLS，不采用历史PASS代替。

| 实际命令/操作 | 结果 |
|---|---|
| CodeGraph、`rg`、`cat`、`sed`、`git status --short --branch` | 先追溯调用和阶段约束；CodeGraph未覆盖的文件再直接读取；分支和开发范围不变 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py check` | 09:00:17–09:00:35 UTC通过；20×130真实、20×22合成组合的search/GET/mget以及120项写入/提权拒绝、TLS/许可门槛通过 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_api.py`（S5入口） | 09:01:48–09:02:27 UTC，1,127次真实HTTP通过 |
| `node --check src/app.js`、`uv run python -B build.py` | JS语法通过；多次构建生成三个静态资源和`S5_BUILD.json`；初始HTML 5,665字节、零内嵌文档 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v`（初轮） | 暴露S2导入脚本对build.py搜索路径副作用的依赖，报`ModuleNotFoundError: es_local`；修正显式导入 |
| 同一Python回归命令（修正后） | **58项通过，19.189秒，退出码0**；包含原57项及新增静态服务检查 |
| `uv run python -B -m unittest discover -s tests -p 'test_s5_static.py' -v` | 最后单独复核静态白名单与无Set-Cookie断言，1项通过 |
| `node --test tests/test_s5_client.mjs` | 最终 **6项通过，退出码0**；不依赖浏览器取消是否成功，验证迟到成功/401、最新查询、raw/download取消、版本不符及读取body中途切换 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080` | 成功启动同源页面/API，仅监听127.0.0.1；日志写入受忽略的`.local/s5/api.log` |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_api.py`（新增静态/菜单字段后） | 最终09:18:05 UTC完成，**1,127次真实HTTP通过**；20身份菜单集合、720次读取、80并发及160次V1问题检查保持通过 |
| Browser实际页面操作、DOM/截图、CDP Network/受限Fetch响应注入 | 结果见`S5_BROWSER_CHECKS.json`；不是旧`set_content`离线页测试。注入仅在测试标签页的特定API响应，未停用ES、修改许可或放宽DLS |
| `uv run python -B -`（只读完整性与凭据检查） | 首次维护脚本遗漏`index_definition(synonyms)`参数，补齐后09:33:00 UTC通过；受保护文件、render源码、ES记录/分析器/同义词及浏览器下载字节一致，见`S5_INTEGRITY.json` |
| `git diff --check`及文档/资源/阶段计数检查 | 用于最后的格式、链接、构建资源与阶段范围复核；Git diff不能覆盖本来未跟踪的整个子目录，另检查本轮文件 |

uv首次在沙箱中访问既有缓存被拒，按工具权限流程在沙箱外重跑，没有另建Python环境或安装新包。浏览器自动化还遇到一次布局采样过早、折叠面板导致定位失败、一次下载事件等待超时；调整等待/真实点击后重新核对。回到首页展开检索面板、窄屏关闭目录时使用inert，均已修正。详情或下载的网络失败不被当作空结果或本地回退。

浏览器与HTTP证据按边界分别记录：

浏览器报告共44条记录：**43条通过，1条原生下载URL导航未验证**。其中下载按钮的HTTP和字节验证为通过，但其操作系统保存步骤仍未验证；两者没有混为同一结论。

| 验证范围 | 实际结果 |
|---|---|
| 初始页面与网络 | 匿名首页只有`/`、`/style.css`、`/app.js`、profile标签与me请求；没有正文。完整示范网络共9次请求，只有1次Document25详情请求，全部同源；见`S5_NETWORK.json` |
| 真实五类身份 | Partner-NCE可见102、Public76/受控26；Partner-OSPA104、Public76/受控28。合同问题分别首项25/99，不出现另一篇受控合同。MSFT、21V、双业务Partner均130，合同结果含25/99 |
| Public跨业务 | NCE打开OSPA Document81；OSPA打开NCE Document3，均经当前cookie和对应native用户读取 |
| 阅读体验 | 结果点击定位菜单并打开正文；English/双语/raw仍属于同一Document。148双语渲染2张表、696单元格，保留rowspan=8的源结构，不重写源表。缺失附件打开引用说明，不请求外部资源 |
| 搜索状态与XSS | loading、成功、真实零命中、503、504及重试分别验证。标题/摘录注入img/script/b标签后仅显示文本，无新增可执行节点；没有虚构回答 |
| 迟到响应 | 暂停成功的OSPA搜索后切NCE、暂停NCE详情/raw/download后切OSPA，旧内容均不恢复；被取消的请求在浏览器被终止。另直接注入旧OSPA版本的成功响应，验证版本守卫也拒绝；Node检查故意忽略AbortSignal补充验证 |
| 当前问题 | 延迟旧问题后提交真实无结果问题，旧成功不覆盖新空结果；菜单名称筛选不发全文检索请求 |
| 会话与历史 | 切换同步清空问题、结果、正文和元数据。复制链接、新标签页、刷新、后退/前进均按cookie重新读取；另一标签页切NCE后原来的99/148深链变为通用拒绝；跨标签页注销清空两页 |
| 下载 | 按钮经API返回200与attachment头；99英文308字节，SHA-256逐字节等于SQL/S2原文。720次HTTP读取包含允许/禁止/匿名下载路径。浏览器原生保存与直接导航限制另列未验证，不把按钮提示当作文件已落盘证据 |
| 静态绕过 | 新增HTTP检查覆盖17个禁止路径（旧HTML、SQL、ZIP、fixture、oracle、日志、原文目录、编码/穿越等）、Host/查询参数、文件及目录符号链接；静态请求不调用ES。三个资源无portal-data/canDoc/expected_access/PlaybookDemo/localStorage |
| 视口 | 1440×1000桌面、390×844窄屏；页面宽度不溢出，长表横向滚动限制在表格容器，窄屏目录打开/选文/关闭通过。证据为`S5_SEARCH.png`、`S5_DESKTOP.png`、`S5_NARROW.png` |

实现依据：[Fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)、[BroadcastChannel](https://developer.mozilla.org/en-US/docs/Web/API/Broadcast_Channel_API)、[CSP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy)。实测结论来自本轮API/浏览器结果，不将这些机制文档当作PASS证明。

## S5：哪些退出条件已经满足

| Roadmap条目 | 结论 |
|---|---|
| S5-01/02/03 | 原源码与阅读样式上改造；专用应用壳，无内嵌正文/权限oracle，真实同源HTTP按需读取 |
| S5-04 | 身份选择、目录与计数均来自API；浏览器不计算授权，没有全库前端回退 |
| S5-05/06/07 | ASK AI状态、最近问题、证据卡/来源版本、目录定位和名称筛选已验证，Network可见真实检索 |
| S5-08 | 双语同一ACL、安全HTML表格、原文、附件缺失说明及文本转义通过 |
| S5-09 | 同步清空、取消、请求/身份/会话版本守卫均有检查；旧正文/原文/元数据/搜索不会恢复 |
| S5-10 | 历史、深链、复制URL、刷新、跨标签页、错误已在页面运行；下载URL的授权与字节在真实HTTP运行，浏览器按钮请求通过。原生下载交互限制明确保留 |
| S5-11 | 固定三个静态资源；禁止路径与符号链接检查通过；CLI自检和权限oracle未下发页面 |

S5的M3退出门槛满足：Network真实API、初始HTML无正文、切换无旧内容恢复、25/99和Public对照可演示、桌面/窄屏阅读保留。此结论不代表S6相关性/性能或生产认证完成。

## S5：哪些仍未验证

- 内置浏览器的原生下载事件等待超时；直接导航下载URL返回`net::ERR_BLOCKED_BY_CLIENT`。**没有确认操作系统保存对话框或文件落盘**，也没有改CSP/浏览器保护绕过。API attachment/字节/权限边界、页面下载请求和会话取消已验证；真实用户浏览器保存流程须补测。
- 测试限于当前内置Chromium与两种视口；没有实机手机、Safari、Firefox、触摸手势或完整辅助技术验收。不将视口模拟称为手机实测。
- 32条问题已获业务审核，身份级相关文档分级、无答案集和质量阈值仍未冻结；没有Hit@5、MRR、nDCG或p50/p95达标结论。
- S6正文/ACL/用户授权更新、撤权确认窗口、撤回、删除、批次切换、回滚、失败恢复及容器重启未执行。前端本轮保护会话/请求竞态，不声称管理变更期间有原子快照。
- 自然许可到期、证书轮换、实际ES中断没有制造，本轮错误测试是受控传输/浏览器响应注入。上游发布状态和时区仍待核验，生产认证/持久会话/远程部署不在本轮。

## 下一阶段入口：S6

用户授权后只推进S6。先复查当前S3/S4门槛、实验开放快照和trial有效性，然后：

1. 以用户已审核的32条V1问题为输入，冻结逐身份/快照的相关文档等级、无答案样例与验收目标；参考ID不自动当作完整金标。权限oracle继续独立，不为通过而改预期。
2. 明确完整快照、差异报告、正文与ACL版本、实验开放迁移和撤权确认窗口，实施最小手工同步；不凭部分数据缺失自动删除文档。
3. 在隔离验证数据/批次中运行更新、ACL变更、用户业务撤权、撤回、删除、失败重试、回滚与重启，证明旧下载/物理索引不能绕过且正文回滚不恢复已撤销ACL。
4. 记录真实质量和p50/p95条件，补齐原生下载保存与端到端验收，分别报告离线、ES、HTTP与浏览器结果。

S5结束，不执行S6。ES容器保持原状态；本轮18080同源服务保留运行，便于打开页面检查。关闭后可按README的uv命令重新启动；单进程重启会使所有mock会话失效。

## S5 补充：ASK AI 弹窗

2026-09-29，按用户新增的页面交互要求调整S5；本轮没有推进S6。

### 实际修改了什么

- `src/template.html`将阅读区的内联检索面板移为原生`dialog`，新增header中的ASK AI入口、弹窗标题和关闭按钮；搜索表单与结果卡继续使用现有API。
- `src/style.css`在桌面居中显示入口，1000px及以下显示右侧悬浮入口；弹窗内独立滚动，窄屏不横向溢出。核对会话期间同时隐藏弹窗中的私有内容。
- `src/app.js`接通点击与`⌘ / Ctrl + K`，原生Esc及关闭/遮罩操作；点击结果立即关闭弹窗，再授权读取文档、更新URL、选中目录并聚焦阅读区。普通关闭保留当前会话最近的问题与结果；切换/注销关闭并清空弹窗，继续使用原有请求和会话版本守卫。
- 重新构建`web/`三个资源和`reports/S5_BUILD.json`，更新README与浏览器状态契约；新增`reports/S5_ASK_POPUP.json`及四张入口/弹窗截图。没有增加依赖，没有改动API、ES索引、ACL、SQL、同义词或验证问题。

### 实际运行了哪些命令和测试

工作目录仍为`playbook-mvp/`，沿用uv环境与已经运行的127.0.0.1:18080同源服务。

| 实际命令/操作 | 结果 |
|---|---|
| CodeGraph调用关系与源码查询，未覆盖部分使用`cat`、`sed`、`rg`读取 | 确认搜索、会话清理、导航与弹窗共用路径 |
| `node --check src/app.js` | 通过 |
| `node --test tests/test_s5_client.mjs` | 6项通过，含迟到响应、身份切换、取消与session_revision守卫 |
| `uv run python -B build.py` | 通过；生成三个资源，HTML 6,005字节、内嵌文档0 |
| `uv run python -B -m unittest discover -s tests -p 'test_s5_static.py' -v` | 1项通过，0.644秒；真实HTTP静态白名单和响应边界仍通过 |
| `lsof -nP -iTCP:18080 -sTCP:LISTEN` | 确认原服务仍仅监听127.0.0.1，未另起服务 |
| 内置Browser实际点击、键盘、DOM与截图 | 21项通过；1440×1000、1024×768、390×844。控制台无warning/error；已关闭辅助标签页并恢复视口 |
| `git diff --check`、`python3 -` / `uv run python -B -`资源指纹、源文件/构建产物一致性检查 | 最终通过；原子目录未跟踪，额外检查本轮文件的行尾空白。首轮截图文件头检查发现工具原生输出JPEG，已将本轮四张截图扩展名改为`.jpg`，没有转换图片或修改旧证据 |

浏览器检查最初有3次测试条件修正：等待身份切换完成后再验证输入焦点；原生弹窗Shift+Tab可进入浏览器工具栏，改为验证原生modal状态及弹窗内Tab顺序；关闭第二测试标签页后对交付页重设窄屏尺寸。均保留在补充报告的`test_attempt_corrections`，最终21项通过，不将初次未满足的断言记成通过。

最终uv一致性核查因既有缓存访问被沙箱拒绝，按权限流程在沙箱外重跑后通过；没有安装新依赖。

### 哪些退出条件已经满足

- 桌面入口精确居中，1024px不与两侧控件重叠；390px悬浮入口可用，弹窗和结果在视口内滚动。
- 点击、⌘K、Ctrl+K打开并聚焦问题；Esc、关闭按钮、遮罩关闭；原生模态及键盘顺序有效，与说明弹窗不叠加。
- NCE合同搜索首项25且不含99，OSPA首项99且不含25。桌面与窄屏点击结果均关闭弹窗并进入对应文档，URL、目录选中项和阅读区焦点正确。
- 同会话重开保留最近问题/结果；跨标签页切换与注销立即关闭并清空弹窗，原深链继续经后端授权检查。匿名状态可以打开入口但不能提交检索。
- S5退出条件继续成立；本次仅验证上述UI调整，没有把此前S3/S4/S5整套历史结果声明为本轮重新运行。

### 哪些仍未验证

- 本轮使用内置Chromium，没有实机手机、软键盘、Safari、Firefox或完整辅助技术验收。
- 原生下载保存/落盘仍为S5原有未验证项；本次没有重新运行58项全量Python回归或1,127次HTTP矩阵。
- S6相关性/性能、正文和ACL更新、撤权、回滚与故障恢复仍未执行。

### 下一阶段的入口是什么

仍为上文S6入口：用户授权后先复查S3/S4实时门槛、实验快照和trial有效性，再冻结逐身份相关性评价与同步/撤权验收。当前同源服务保留运行，页面可直接试用ASK AI新入口。

## S6：实际修改了什么

本轮仅实施S6，分支仍为`playbook-local-search-mvp`，范围仍为`playbook-mvp/`。没有自动进入生产阶段、提交或推送。使用原uv环境，没有新增Python/npm依赖。

| 文件 | 实际变化 |
|---|---|
| `evaluation/golden_v1.json`、`S6_GOLDEN_REVIEW.md` | 用户明确确认“确认草案与 90% 目标”后，冻结32条逐身份参考、3条无答案样例和每类身份Hit@5≥90%；原问题集未改写 |
| `tools/evaluate_search.py` | 审核稿/批准冻结、指纹校验、真实用户HTTP检索、Hit@5/已标注Recall@5/MRR@5、无答案返回情况、ES/API性能和硬件证据 |
| `runtime.py`、`api.py` | POSIX本机读写锁、持久化维护门禁、原子catalog发布、请求绑定snapshot和generation、成功更新后旧会话401；验证页可用固定`--validation`隔离配置 |
| `tools/sync_playbook.py` | 六表完整快照静态导出、正式关系/身份检查、差异与不可改写计划、显式开放/删除、门禁内批次更新与全量比对、native授权更新、同计划失败重试、只回滚正文 |
| `tools/ingest_playbook.py`、`tools/es_permissions.py` | 复用既有导入/DLS能力；增加S6接管后拒绝旧初始化/导入/开放覆盖权限的保护；凭据和DLS支持固定隔离命名空间 |
| `tools/check_s6.py`、`tests/test_s6.py` | 独立合成ES/HTTP变更矩阵与8项离线保护；覆盖进程中断持久门禁、过期许可、完整性、金标篡改、回滚审核期间再撤权等 |
| `src/app.js`、`src/template.html`、`web/`、`tests/test_s5_client.mjs` | 可见页15秒后台核对会话、失败清空旧内容；修复周期核对抢输入焦点；无答案提示要求查阅原文，保留header/浮动入口、原生popup和快捷键 |
| `tools/check_mvp.sh` | 串联现有离线、构建、DLS、API、金标和隔离变更命令，浏览器证据单列 |
| `reports/S6_*`、`reports/screenshots/s6-ask-ai.jpg` | 评测、变更/失败尝试、浏览器、完整性、一次浏览器撤回操作及截图证据 |
| `README.md`、`CONTRACTS.md`、`AGENTS.md`、Roadmap及本台账 | 同步/恢复操作、人工窗口、回滚保护、冻结目标、阶段结论、未验证项与后续入口 |

完整性实测：原SQL、抽取器、fixture生成器、`mock_auth.py`、旧HTML和正文render函数均与S5指纹一致，三组同义词未变；真实130条完整ES记录指纹仍为`e730f69e85a969f4cca1fc185aaa3bcc65b304ccf7b220553afc8b697318ecd5`。20个业务账号的DLS/metadata复查通过。本次真实变更只发生在明确合成的`playbook-s6-validation`、旧副本`playbook-s6-archive`和`pb_s6_*`，不据合成差异伪造真实角色ACL差异。

## S6：实际运行了哪些命令和测试

目录为`/Users/claus/Hybrid-Search-Engine/playbook-mvp`。所有需要mock的Python命令使用`APP_ENV=test MOCK_AUTH_ENABLED=1`。uv初次访问缓存受到沙箱限制后按权限流程重跑，未绕过缓存或安装新依赖。

| 实际命令/动作 | 实测结果 |
|---|---|
| `uv run python -B tools/es_permissions.py check` | 阶段入口和容器重启后各运行一次，最终20身份×130真实ID、20×22合成ID及120项禁止操作全部通过；含真实search/GET/mget、匿名401和TLS反例 |
| `uv run python -B tools/check_api.py` | 入口及重启后各1,127次真实HTTP，通过720次详情/原文/下载、80次并发请求、160次V1问题输入及会话/静态/日志边界 |
| `uv run python -B tools/evaluate_search.py prepare-review` | 生成审核稿，未使用本轮排名反推标注 |
| `uv run python -B tools/evaluate_search.py freeze --approved-by user:current-thread --evidence '确认草案与 90% 目标'` | 根据用户明确确认执行一次；冻结指纹`81005463989a950e1bb689bcea4daf0dfe7485e8787d2025d8c70a261276e118` |
| `uv run python -B tools/evaluate_search.py run` | 五类身份全部达到90%目标；25次暖机、480次测量、15次无答案组合，原金标未改动，检索配置未调参 |
| `uv run python -B tools/check_s6.py`、`uv run python -B tools/check_s6.py --restart-es` | 最终带重启运行通过：737次HTTP、199次用户ES检查；试验内真实部分写入后注入失败，门禁/reader拒绝成立，同计划重试恢复；尝试历史在`S6_CHANGE_RUNS.json` |
| 脚本内`docker compose restart es` | 保留数据卷；实测503/401恢复到重新登录后的200，记录/撤权状态持久，旧物理索引依旧拒绝 |
| `uv run python -B -m unittest discover -s tests -p 'test_s6.py' -v` | 初版7项通过；后增加回滚包生成后再发生权限变化的拒绝用例，由最终全量回归覆盖 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 从阶段入口58项到初版65项，再到最终**66项全部通过**；最后运行32.359s |
| `node --test tests/test_s5_client.mjs` | 原6项及新增后台核对不禁用输入/失败清理用例，最终**7项全部通过** |
| `uv run python -B build.py` | 多次随页面修正构建；最后只有3个静态资源，嵌入文档0，源码/产物一致 |
| Browser实际页面操作、只读DOM检查和截图 | **9项S6检查通过**：隔离授权文档/弹窗、远程撤回后的清空、重新登录仍拒绝、真实无答案提示、NCE命中25排除99、结果跳转、最终后台焦点保持、最终popup搜索和⌘K；见`S6_BROWSER_CHECKS.json` |
| `uv run python -B -`浏览器变更脚本 | 在独立合成索引撤回99001，成功写入1条、维护2864.96ms；浏览器定期核对后清空旧内容。证据`S6_BROWSER_MUTATION.json` |
| `uv run python -B -`完整性/资源核查 | 源文件、AST函数段、同义词、真实索引和20身份不变；容器2 GiB，CPU未设置单独quota，主机16 GiB、8逻辑核；写入`S6_INTEGRITY.json` |
| `bash -n tools/check_mvp.sh`、`git diff --check`及本轮文件行尾检查 | 语法/空白检查通过；串联脚本各条命令已分别运行，未声称整份shell脚本另跑一遍 |
| `api.py --port 18080`、`api.py --validation --port 18081`及已确认PID的TERM | 主服务重启加载S6代码，最终继续运行；临时18081隔离页面服务在验收后关闭，数据/凭据保留用于重现 |

本轮保留了失败和修正经过，不将初次失败算作通过：第一轮误期望跨索引mget外层403，实际是200内逐项security_exception，修正为检查每项拒绝且无正文；第二轮ES重启出现401后原cookie失效，修正为恢复后重新认证；最终带重启运行通过。页面定期会话检查最初会使输入失焦，已改为后台核对并增加客户端检查，重载构建后的页面验证焦点保持。完整性首轮使用`inspect.getsource`的尾换行与旧算法不同，改用S5相同AST source segment后指纹完全一致。一次额外query参数导航被浏览器客户端拦截，原应用URL和index.html正常打开。

## S6：哪些退出条件已经满足

- 用户批准并冻结V1；原32条问题、参考ID和三组同义词不变。五类身份按同一已开放源快照计分，11个无可访问参考的组合单列排除，不算漏召回。Q23为间接外链等级1；未标注资料仍为未知。

| 代表身份 | 有可访问参考的问题 | Hit@5 / 已标注Recall@5 | MRR@5 |
|---|---:|---:|---:|
| MSFT | 32 | 31/32 = 96.875% | 0.8698 |
| 21V | 32 | 31/32 = 96.875% | 0.8698 |
| Partner NCE | 27 | 26/27 = 96.296% | 0.8642 |
| Partner OSPA | 26 | 26/26 = 100% | 0.9423 |
| Partner 双业务 | 32 | 31/32 = 96.875% | 0.8698 |

- 唯一核心未命中是Q32“ NCE 更换合同合作伙伴 COCP 有什么流程？”的参考Document18，在四类可访问身份下未进入Top5；保留为V2改进输入，没有改标注或为通过阈值调整检索。Recall与Hit数值相同是因每题只有一个已标注参考，不表示相关文档已穷尽。
- 有真实性能报告：单并发480次、暖机25次；ES took p50=8/p95=19ms，完整HTTP p50=46.648/p95=58.398ms；Intel i5-1038NG7、macOS15.7.5、16 GiB、ES容器2 GiB/堆1 GiB。Docker限制在测量后单独核查并记录时间，不把后补核查伪装成测量时采样。
- 无答案3×5组合返回率100%。页面明确结果为文档/摘录并要求原文确认，未生成“答案”、未设置未标定阈值或跨权限补足K。该指标仅描述当前局限，没有另造通过目标。
- 完整快照/关系校验、独立ACL指纹差异、显式开放/删除、幂等、维护门禁、全量后验比对、代次失效、失败重试与旧命令保护已实现。真实隔离变更后新API/ES请求只看到当前授权；正文回滚不恢复已撤ACL、业务范围、撤回及已删除ID，旧索引/raw/download不能旁路。
- 成功确认后的请求边界与人工同步窗口已写入契约。已开始的API读取持有共享锁至响应发送完成；更新排空后发布。可见页按15秒周期核对，失败清空；不是网络端到端硬实时保证。前端旧响应保护和服务端会话保护均有检查。
- 日志只记录既定字段，mock生产禁用、匿名、原文净化/XSS资源允许列表、普通用户写入拒绝继续通过。真实DLS/HTTP重新运行，未用离线PASS替代。
- 原源数据、真实索引和业务账号未变，M4的本地验收范围完成；没有启用生产认证、LLM、向量或Agent。当前页面：[Playbook](http://127.0.0.1:18080/)。

## S6：哪些仍未验证

- 主130篇业务索引未进行新的真实业务快照切换；同步与回滚的真实ES变更实测限定于合成隔离环境。真实六表导出/转换完整性已离线验证，不等同生产上游事务一致性验证。
- 缺少真实业务用户独立提交的测试集、穷尽相关性标注/总体召回率及独立留出集；当前35题不能证明泛化效果。无答案识别尚无可靠策略。
- 生产登录/外部授权、上游发布语义及状态枚举、CDC/自动同步、多主机锁与分布式一致性、并发压力和性能SLA未实现/未验证。
- 许可自然到期和证书轮换未实测；过期许可拒绝同步及查询失败处理为离线/故障注入证据，实际trial当前有效至2026-10-29T07:27:15.030Z。进程异常退出门禁已用真实子进程验证；ES部分写入中断为注入，不冒称断电事故。
- 本轮9项Browser检查不能替代整套S5历史浏览器矩阵、Safari/Firefox、实机手机/软键盘、完整辅助技术或原生OS下载保存验证。远程撤回清理在焦点调整前已实测，最终quiet模式清理另有客户端回归，最终页面实测保持焦点。
- 回滚审核期间源权限再次变化的新增保护通过最终离线回归；没有另制造第二轮真实并发撤权事故。串联shell入口通过语法检查，各组件已运行，未单独重复整份脚本。

## S6：下一阶段的入口是什么

S0–S6在本地范围结束，不自动启动下一阶段。后续入口是：收集真实业务问题并审核V2（包括Q32与无答案策略），再由用户单独确定生产认证/用户授权、上游发布语义及完整导出/撤权时效协议。新增同义词继续走业务确认和版本化；新快照须重建并审核对应权限/相关性预期，不能修改V1让报告继续通过。

## S0–S6 复查与代码库清理（2026-09-29）

### 实际修改了什么

本轮按用户要求复查已完成的 S0–S6 并清理中间产物，没有追加产品阶段。复查范围是当前 `playbook-mvp/` 全部实现；目录仍未纳入 Git，仓库 HEAD 为 `e6629fd1a96cc971a331f5866d37796f517e95bf`，不能以空 Git diff 代替源码复查。

| 文件/范围 | 实际变化 |
|---|---|
| `README.md` | 从阶段开发记录改为当前 MVP 指南：运行要求、已有环境启动、首次初始化、显式实验开放、页面使用、验证、数据同步/恢复、故障排查和当前限制 |
| `.gitignore` | 补充 uv 环境、Python 缓存、`.DS_Store` 与可重建的 `web/`；继续忽略含凭据和恢复状态的 `.local/` |
| `tools/check_mvp.sh` | 将页面构建移到测试之前，使缺少 `web/` 的工作目录也能直接执行自动验收 |
| `requirements-build.txt`、`requirements-test.txt`、`tests/browser_check.py` | 删除重复的 pip 依赖清单与只面向旧全量 HTML 的 Playwright 脚本；当前依赖继续由 uv 管理 |
| `CONTRACTS.md`、`docs/CODEX_HANDOFF.md` | 澄清兼容副本仍用于 S0 回归；原包交接说明标为来源记录，并指向当前运行指南 |
| `reports/S3_CHECKS.json`、`S4_CHECKS.json`、`S6_EVALUATION.json`、`S6_CHANGES.json`、`S6_CHANGE_RUNS.json` | 更新为本轮实际自动验收结果，变更检查历史继续保留 |
| 本台账 | 记录复查结论、清理明细、实际命令及验证边界；历史阶段记录保留在此 |

共删除 19 个过时入口或临时文件，其中上述 3 个旧入口，其余 16 个共 1,042,181 字节：

- `.DS_Store`。
- `.local/s5/` 内 9 项：`api-regression.log`、`api.log`、`build.log`、`client-tests.log`、`first.png`、`icons.js`、`narrow-first.png`、`render.sha256`、`unittest.log`，随后移除空目录。
- `.local/s6/browser-validation.log`。
- 原离线浏览器草稿产物：`reports/desktop_preview.png`、`document_preview.png`、`mobile_preview.png`、`users_preview.png`、`download_roundtrip_99.md`。

删除前逐项核对文件类型及进程占用。原始 SQL、抽取器、fixture 生成器、独立 oracle、32 条问题、冻结金标、同义词和 31 份凭据未变。保留原 HTML、`data/portal_data.json`、兼容副本、MANIFEST 及实际验收报告：其中 S0 测试依赖独立正文/来源对照、CLI 场景和重建报告，S3 检查还读取 `S2_RELEASES.json` 的开放批准，不能作为无用产物清除。

`web/` 的三个文件曾完整删除以验证重建，验收脚本随后生成与 `src/` 逐字节一致的资源。保留 `.local/s2` 导出及开放记录、S6 历史计划/决定/批次和 Docker 数据卷；本轮隔离变更测试正常产生新的验证批次，不清除恢复依据。

### 复查结论

- **Standards**：复查角色规范化、请求专属 native 凭据、完整 DLS、静态白名单、前端代次保护及同步门禁/回滚，未发现已确认的阻断缺陷。没有为代码风格引入额外抽象或重构权限实现。
- **Spec**：S0–S6 当前实现与本地验收范围一致；修正 README 中过时的阶段统计与启动说明。首次初始化和已有环境启动分开，开放仍需独立显式命令。新增的构建顺序保证生成文件不入库后仍可运行验收。

### 实际运行了哪些命令和测试

以下项目在 `playbook-mvp/` 实际执行；uv、Docker 和本机端口访问遇到沙箱限制时按权限流程运行。

| 命令/动作 | 实际结果 |
|---|---|
| `bash tools/check_mvp.sh` | 从缺少 `web/` 的状态完整执行，退出码 0；本轮确实运行了整份脚本 |
| 脚本内 `uv run python -B build.py` | 重建三个资源，无内嵌文档，源码与产物一致 |
| 脚本内 `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 66 项通过，30.797 秒 |
| 脚本内 `node --test tests/test_s5_client.mjs` | 7 项通过 |
| 脚本内 `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/es_permissions.py check` | 20×130 真实文档和 20×22 合成组合、120 项禁止操作、匿名及 TLS 边界通过，业务数据不变 |
| 脚本内 `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_api.py` | 1,127 次真实 HTTP 检查通过 |
| 脚本内 `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/evaluate_search.py run` | 五类身份 Hit@5 仍为 96.296%–100%，全部达到冻结 90% 目标；480 次测量的 API p50/p95 为 48.201/59.926 ms，ES took 为 9/19 ms；无答案返回率仍为 100% |
| 脚本内 `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_s6.py` | 646 次 HTTP、187 次用户 ES 检查通过；包含部分写入失败后重试、保留撤权的正文回滚及回滚后禁用账号；主业务索引未变 |
| `uv sync --locked` | 锁文件解析与现有依赖检查通过，无新增依赖 |
| `docker compose config --quiet`、`docker compose ps --all`、`docker compose up -d --wait --wait-timeout 180` | 配置有效；已有 ES 保持运行，证书准备服务正常退出，ES healthy；未删除卷或重新批准开放 |
| `APP_ENV=development MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080` | 按 README 启动主服务，随后完成真实 HTTP 冒烟检查 |
| `uv run python -B -` 启动冒烟 | 首页等于构建产物；NCE 身份搜索“合同的生成”返回 `[25,24,27,89,15]`，25 可读、99 返回404；注销204，注销后读取返回401 |
| `uv run python -B -` 完整性核对 | 清理前采样的151个文件中150个逐字节不变，唯一变化是正常测试更新的隔离 `active.json`；31份凭据保持600且被Git忽略，主索引未被S6接管；README的10个bash代码块语法与本地链接有效 |
| `bash -n tools/check_mvp.sh`、`git diff --check`、文档/脚本行尾与忽略规则检查 | 通过；对尚未跟踪的文件另行检查，不以空diff充当有效验证 |

冒烟脚本首轮把注销成功误期望为200，实际API正确返回204；修正检查脚本后完整通过，未修改API行为。完整自动验收本身一次通过。本轮报告时间为UTC 2026-09-30 02:17–02:20，对应本地2026-09-29。

### 哪些退出条件已经满足

- README 已可作为当前 MVP 的启动和使用入口，历史开发过程集中保留在台账。
- 已清除确认无运行/回归依赖的中间产物，凭据、独立基线和恢复资料未丢失。
- 从无页面构建产物到完整自动验收实际通过；README 日常启动及授权检索链路已实测。
- 原业务索引、SQL、冻结金标、同义词和账号凭据保持不变；未调整预期换取通过。

### 哪些仍未验证

- 本轮未新建空 Docker 数据卷重跑首次初始化；首次命令按CLI和既有阶段证据复核，现有环境日常启动已实测。
- 本轮没有执行 `--restart-es` 或重跑浏览器交互矩阵；此前带重启的737次HTTP/199次用户ES检查保留在变更历史，浏览器记录仍为原阶段证据。
- 生产认证、主业务索引的新业务快照切换、真实业务独立问题集、自然许可到期及证书轮换等，继续沿用S6未验证范围。

### 下一阶段的入口是什么

按 README 使用当前 MVP。后续功能或生产接入仍由用户单独确定；本轮不自动进入新阶段。主API保留在 `http://127.0.0.1:18080/` 供本地使用。

## 搜索结果补充：相关性得分与匹配高亮（2026-09-29）

### 实际修改了什么

- `api.py`：在原有最多 5 条结果中返回 Elasticsearch 原始 BM25 `score`，显式指定分数降序、文档 ID 升序处理同分；检索词、字段权重和权限过滤不变。标题及摘录通过 Elasticsearch 高亮器产生匹配信息，先对原文作 HTML 编码，再转换为纯文本与 `highlighted` 标记；原文中的标签保持文字。摘录按解码后的 Unicode 字符截断到 300 字。
- `src/app.js`、`src/style.css`、`src/template.html`：每条结果显示保留两位小数的“相关性得分”，标题和摘录中的匹配词以浅黄色高亮。页面沿用服务端顺序，并说明分数只用于同一次检索比较，不代表答案正确率。前端逐片段转义文本后生成固定的 `<mark>` 元素。
- `tests/test_s4_api.py`、`tests/test_s5_client.mjs`：新增分数、高亮及恶意 HTML 原文的回归检查，包括 Unicode 截断、异常分数拒绝、服务端排序保留和不完整高亮数据退回纯文本。
- `tools/check_api.py`：真实 HTTP 验收增加有限非负分数、降序/同分顺序及高亮片段与原文一致性检查。
- `CONTRACTS.md`、`README.md`：更新接口字段和当前页面使用说明；README 不添加开发过程记录。
- 重建 `web/`，更新 `reports/S5_BUILD.json`、`reports/S4_CHECKS.json`；新增 `reports/SEARCH_DISPLAY_CHECKS.json` 和 `reports/screenshots/search-score-highlights.jpg` 保存本轮浏览器证据。

### 实际运行了哪些命令和测试

下列命令在 `playbook-mvp/` 执行，Python 继续使用 uv 管理的环境。

| 命令/动作 | 实际结果 |
|---|---|
| `uv run python -B build.py` | 成功生成 3 个静态资源，嵌入文档数为 0 |
| `uv run python -B -m unittest discover -s tests -p 'test_s4_api.py' -v` | 7 项通过，12.674 秒 |
| `uv run python -B -m unittest discover -s tests -p 'test_*.py' -v` | 67 项通过，44.928 秒 |
| `node --test tests/test_s5_client.mjs` | 8 项通过，包含原有会话/过期响应保护和新增安全高亮渲染检查 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B tools/check_api.py` | 20 个模拟身份、1,127 次真实 HTTP 检查通过；新分数和高亮字段检查覆盖实际返回结果 |
| `APP_ENV=test MOCK_AUTH_ENABLED=1 uv run python -B -` 排名与金标对照 | 本轮 HTTP 报告中 160 组身份/问题的 Top 5 文档及顺序，与已有 S6 评价报告完全一致；冻结金标指纹仍为 `81005463989a950e1bb689bcea4daf0dfe7485e8787d2025d8c70a261276e118` |
| 进程命令和工作目录核对、`TERM` 旧 API、`APP_ENV=development MOCK_AUTH_ENABLED=1 uv run python -B api.py --port 18080` | 仅重启已确认的本项目 API 以加载新代码；新服务保持运行供页面使用 |
| Browser 真实页面操作、只读 DOM 检查和截图 | 7 项通过：中文分数降序与高亮、点击结果跳转、英文短语高亮、空结果清理、用户切换清理、OSPA 授权结果及高亮、⌘K 打开弹窗；浏览器未记录 error 日志 |
| `git diff --check`、`uv run python -B -` 最终文件核查 | 对未跟踪的修改文件另行检查行尾与末尾换行；生成的 JS/CSS 与源码一致，7 项浏览器证据均通过、截图存在；清除本轮 `/private/tmp` 下两份已完成命令的临时日志 |

浏览器中 NCE 用户搜索“合同的生成”的结果为 `[25,24,27,89,15]`，展示分数为 `[30.74,11.58,10.54,7.00,6.69]`；切换 OSPA 后为 `[99,101,92,89,15]`，受控 NCE 文档 25 不可见。最终页面恢复 NCE 用户和中文搜索示例。英文 `Savings Plan` 的实际分词及已批准同义词匹配同样高亮。自动报告时间为 UTC 2026-09-30，对应本地 2026-09-29。

### 哪些退出条件已经满足

- 每条授权结果显示相关性得分并按服务端原始分数降序排列，最多 5 条；没有将 BM25 分数转换成未经标定的百分比或正确概率。
- 标题和摘录高亮来自 Elasticsearch 的实际匹配，支持中英文、当前分词和已批准同义词；原文 HTML 不作为可执行代码插入。
- 20 身份真实 HTTP 验收通过，浏览器完成 NCE/OSPA 切换验证；权限边界、会话保护及已有检索排序保持成立。
- 160 组 V1 检索结果排序与原验收报告一致；未修改金标、同义词或业务索引换取通过，未添加依赖。

### 哪些仍未验证

- 本轮未重新执行 480 次性能基准；既有 S6 延迟数据属于原运行，不能视为加入高亮后的新性能测量。
- 本轮未重新执行 ES 重启、同步/回滚演练或完整 S5 浏览器矩阵。跨浏览器、手机实机及辅助技术验证仍沿用已有未验证范围。
- 未进行得分到业务正确概率的校准；当前交付的是检索相关性得分。生产认证、独立业务问题集等继续沿用 S6 的边界。

### 下一阶段的入口是什么

在当前 MVP 的 ASK AI 弹窗中体验分数和高亮；后续根据真实业务反馈决定是否调整检索。新增同义词或评价数据仍须业务确认，不自动启动新阶段。
