# Playbook MVP 契约 v1

冻结阶段：S0；S2–S6实现补充：2026-09-29。已实现本地入库、DLS、API、页面和手工同步。兼容字段、身份、API及S6同步/评测约束见下文和验证台账。

## 1. 来源、路径与版本

内容与关系依据为原始 SQL，权限语义依据为用户已确认的互斥角色、内部默认双业务、Partner 显式业务及 Public 例外。派生结果必须能回到 SQL，不能反向补造上游事实。

| 输入 | S0 决定 |
|---|---|
| `data/playbook-sampledata.sql` | 唯一内容/关系源；UTF-16；只做静态解码 |
| `tools/extract_playbook_sql.py` | 唯一现有 SQL 抽取器；原包只复制了此脚本，原文、metadata、ACL 可在新目录重建 |
| `fixtures/mock_users.json` | 唯一业务身份 fixture，schema 2.0，20 个预设 profile |
| `fixtures/mock_users_effective.json` | 身份规范化的可比较输出，不是浏览器可提交的授权凭据 |
| `fixtures/document_acl_snapshot.json`、`source_acl_tables.json` | SQL 关系的派生快照与来源证据 |
| `fixtures/expected_access.json`、`authorization_cases.jsonl` | 独立 SQLite JOIN 得到的逐 ID 权限 oracle，禁止用作检索实现 |
| `fixtures/es_metadata_candidates.json` | 原包的未部署metadata投影，保留不变用于独立对照；S3实际部署见reports/S3_* |
| `data/mock_users*.json`、`data/expected_access.json` | 离线参考兼容副本，由 S0 回归核对与 `fixtures/` 一致；在线服务不从这里建立第二套身份源 |
| `src/`、`build.py`、生成 HTML、`data/portal_data.json` | UI 参考与离线原文对照，不是在线授权来源；不能通过 HTTP 下发全量载荷 |
| `MANIFEST.json`、`docs/CODEX_HANDOFF.md` | Mock v2 的原始包记录；MANIFEST 不代表合并后的整个项目，README 已改为项目文档 |

源实例命名空间固定为 `playbook-test`，这是本地配置，不冒充上游数据库名。SQL 的 SHA-256 固定为：

```text
cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821
```

本快照：Document 130、MenuItem 164、MenuItem_Document 130、Role 3、Role_MenuItem 195、Notice 4。Notice 不并入知识文档。文档 ID 非连续，不能把 `1..130` 当成文档集合。260 个语言源字段必须分别保留；原抽取 ZIP 和 260 份常驻 Markdown 不是 S0 的必需输入。

旧 Excel 的“42 篇 ACL 可解析、88 篇不可解析”、旧 Document 146 映射和 Mock v1 的混合角色规则退役，不参与新版断言。旧 ES Starter、本轮未复制的抽取包 REVIEW、历史浏览器报告不能作为当前环境通过的证据。外部认证 C# 附件仅解释身份来源，本轮不实现其登录协议。

Python 使用 uv；`.python-version` 固定为本机已有的 CPython 3.12.12。S0 无第三方依赖；S2 为正文解析创建 `pyproject.toml`/`uv.lock`，复用原构建器使用的 markdown-it-py、BeautifulSoup4、Bleach 及其既定版本，不引入新的 API/ES SDK。

## 2. 事实、业务规则与实验假设

| 类别 | 内容与处理 |
|---|---|
| SQL 事实 | 130 篇文档各有一个直接菜单关系；Public 76（NCE 43、OSPA 33），受控 NCE 26、OSPA 28；三个角色各有相同的 65 个受控菜单授权 |
| 已确认业务规则 | 角色互斥；MSFT/21V 默认双业务但不豁免菜单；Partner 显式业务；Public 对合法身份跨业务可见 |
| 下游派生 | Markdown 文件、清洗后的检索文本、安全阅读 HTML、规范化身份、ACL、指纹、权限 oracle；不能称为上游发布版本 |
| 本地门槛 | `MenuItem.Active=1`、授权角色 `Role.Status=1`；非 1 不授予相应访问，正式枚举含义仍未核验 |
| 未核验发布语义 | 保留 `Document.Status` 原值及 `publication_state=unverified`，使用独立实验开放记录 |
| 多菜单/继承 | 当前只接受一个正式直接菜单关联；缺失、多挂载、跨业务不一致、未知 UnderControl/业务值先隔离；不推导父菜单继承 |
| 时间与格式 | SQL datetime 不含已确认时区，保留原字符串，`timestamp_timezone=null`，不追加 `Z`；TextFormatType 保留原值，不据它伪造内容 |
| 语言与附件 | ContentEn 不保证纯英文；相同语言字段不删除，空值不回填；附件仅保留引用，不抓取缺失文件 |

## 3. Document 25/99 的来源追溯

行号为原始 SQL 解码后的 1-based 行号，不是生成 Markdown 的行号。

| 证据 | Document 25 | Document 99 |
|---|---|---|
| 标题 | 合同的生成 | 合同的生成 |
| 业务 | NCE=0 | OSPA=1 |
| Document 行 | 2985–3005 | 9689–9697 |
| 正式菜单 | MenuItem 31，行 13269 | MenuItem 114，行 13357 |
| 菜单原值 | Type=0、UnderControl=1、Active=1 | Type=0、UnderControl=1、Active=1 |
| 关系 | MenuItem_Document 87，行 13440：31→25 | MenuItem_Document 685，行 13503：114→99 |
| MSFT 授权 | Role_MenuItem 146，行 13735 | Role_MenuItem 845，行 13820 |
| Partner 授权 | Role_MenuItem 173，行 13762 | Role_MenuItem 878，行 13853 |
| 21V 授权 | Role_MenuItem 200，行 13789 | Role_MenuItem 911，行 13886 |

Role 1/2/3 的定义分别在行 13707/13708/13709，Status 均为 1。Partner-NCE 可读 25、不可读 99；Partner-OSPA 相反；内部用户与双业务 Partner 均通过这两篇的 ACL。未来实际读取还需会话及实验开放，不保证某句自然语言的 Top K 必定命中。

Public 对照为 Document 3→菜单 5（NCE）和 Document 81→菜单 90（OSPA）。OSPA 访问权限说明为 Document 148→菜单 174；Document 146 实际是 NCE Public《Azure 预留交换策略的变化》。

## 4. DocumentRecord 最小契约

一个逻辑 Document 对应一条 ES 记录，中英文为同一记录的字段。菜单不是额外知识文档。下表是规范化后的内部对象，不是全部对外返回的 DTO；S2 物理 ES mapping 见 `documents.index_definition()`。

| 字段 | 类型与语义 |
|---|---|
| `schema_version` | 字符串 `1.0` |
| `document_key` | `playbook-test:Document:<Document.Id>`；稳定主键，不使用标题、语言或快照 hash 作主键 |
| `document_id`、`kb_id` | 源整数 ID；`playbook` |
| `title`、`title_en`、`slug` | 原始 Title、TitleEn、Slug；保持源值及空值，不翻译 |
| `business_category` | 严格整数 0 或 1，不把 bool 当整数 |
| `source` | `instance_key`、`table="dbo.Document"`、`primary_key`、`snapshot_sha256`、`sql_lines=[start,end]`；原文件路径只留服务端 |
| `upstream_metadata` | 除 Content/ContentEn 外的原始业务字段；Status、TextFormatType、时间原值保留，不把 `_sql_lines` 当业务字段 |
| `timestamp_timezone` | `null`，直至有可信上游定义 |
| `languages.zh`、`languages.en` | 每项有 `source_field`（Content/ContentEn）、`raw`（string 或 null）、`raw_utf8_sha256`（null 对应 null）、`search_text`、`headings` 字符串列表 |
| `acl` | 见下节；来自正式关系，不从正文或标题推断 |
| `acl_provenance` | 正式关系、菜单、角色授权及角色行的 ID 和 SQL 行号；保留抽取器已有证据 |
| `publication_state` | 当前仅允许 `unverified` |
| `lab_release` | 见第 5 节，默认未开放 |
| `content_revision` | 复用抽取器 `content_fingerprint`：对原 Content/ContentEn 对象做 canonical JSON SHA-256；不是上游版本 |
| `acl_revision` | 复用抽取器对 `semantic_acl()` 的 fingerprint，包含策略、菜单、业务、正式角色及状态；不是上游版本 |

Canonical JSON 使用 UTF-8、`ensure_ascii=False`、`sort_keys=True`、`separators=(',', ':')`。正文指纹不包含标题和其他 metadata；这些变化仍由 `source.snapshot_sha256` 标识。旧 Portal 对语言列表计算的 `content_sha256` 与抽取器对象指纹算法不同，不能混用。

原文和检索文本分开：原文逐字符及换行保真；搜索文本从 Markdown/内嵌 HTML 提取，保留步骤与表格关联，排除装饰 SVG/CSS。阅读 HTML 是经允许列表清理的派生显示内容，不覆盖原文。两个语言字段相同仍保留；查询两个语言时采用 `dis_max(tie_breaker=0)` 取最高语言分数，不累加重复正文。S2 已验证原生 ES 行为，API 查询与标题/章节权重在 S4 实现验证。

### 4.1 ACL 形状与转换

规范化对象沿用 Mock v2 的形状：

```json
{
  "state": "resolved",
  "menu_id": 31,
  "menu_active_raw": 1,
  "is_public": false,
  "allowed_role_ids": [1, 2, 3],
  "required_business_category": 0,
  "role_status_raw": {"1": 1, "2": 1, "3": 1}
}
```

- `allowed_role_ids` 是正式菜单授权的排序去重集合；访问时还要检查当前角色的 `role_status_raw` 为 1。
- Mock 快照已使用字符串角色 ID→状态的对象。抽取器的 index candidate 使用 `{id,status}` 列表；转换为上述对象。原始 `derive_acl()`/Portal 的状态证据在 `provenance.Role`/`acl_provenance.Role` 中，必须从正式 Role 行读取，禁止缺失时补 1。
- 同一个角色出现矛盾状态、未知角色、缺失关键字段或关系异常时不得标为 resolved。隔离项仍保留来源与原因，不进入用户可读集合。
- `acl_revision` 在转换前取自抽取器原始语义与状态指纹，转换不得丢失状态信息。S2 可以增加等价的索引投影以支持 DLS，但不能省略角色状态门槛或改变 Public 例外。

### 4.2 S2 物理实现与兼容字段

索引固定为 `playbook-documents-v1`，`_id=document_key`；`dynamic=strict`、单分片、无副本。`schema_version=1.0` 保持不变，新增 `transform_version=s2-v1` 标识本地转换版本。以下字段是内部追加，不能直接作为 S4 对外 DTO：

| 字段或配置 | S2 语义 |
|---|---|
| `same_content` | 原 Content 与 ContentEn 完全相同的标记，两个原字段均保留 |
| `languages.*.html`、`assets` | 复用原 `build.render()` 的允许列表 HTML 与附件引用；不运行旧全量构建、不下载附件 |
| `table_count`、`merged_cell_count`、`table_warnings`、`omitted_decorations` | 语言内的解析诊断；超出实际行组的 rowspan 仅展开已有行并记录警告，原文及 HTML 不修写 |
| `acl.active_role_ids` | 正式 allowed_role_ids 中 Role.Status 严格为1的角色；ACL unresolved 时为空。DLS 同时检查 allowed 和 active，不凭空补默认状态 |
| `acl.reasons` | 未知/歧义关系或状态的隔离原因；隔离非空阻止整批导入与开放 |
| 精确字段 | ID/业务/菜单/角色用 integer；源键、指纹、状态用 keyword；公开/开放标记用 boolean |
| 存储字段 | raw/html 为 `index=false`，upstream_metadata/acl_provenance/assets 为 `enabled=false`；保留 `_source`，不把 SQL 包装、时间/状态原值或 SVG/CSS 当知识检索 |
| 分析字段 | title/title_en、languages.*.headings/search_text 使用 smartcn；查询链为 smartcn_tokenizer → porter_stem → smartcn_stop → synonym_graph |

同义词仅消费 `config/synonyms.json` 的已审核规则，索引 `_meta` 保存配置指纹。2026-09-29 用户确认三组只用于本地实验：RI/预留实例、Savings Plan/节省计划、CICR/Contact Information Change Request；不扩展为取消/删除/暂停互为同义词。变更 mapping 或分析器需显式维护，导入不会自动迁移或删除旧索引。

数据由 `documents.load_records()` 复用原 SQL 解码、正式 ACL 推导与指纹算法形成；`.local/s2/<snapshot>/` 保存不可覆盖的原文导出以及可重建派生文件。Bulk 使用稳定 ID 和乐观版本条件，逐项记录错误；完全相同的记录跳过且保留批准，发生变化的记录恢复默认未开放。源删除停止并留给 S6 显式流程，不因新批次少一篇就自动删旧数据。

当前仅支持单一固定 SQL 快照、1,000 篇/10 MiB 内的本地批次。S2 reader 的 DLS 同时固定实际源快照与批准快照，不能直接沿用 S1 的合成快照；未来切换快照及 reader 策略的同步窗口须在 S6 验证。

## 5. 实验开放与读取判定

新记录的默认值：

```json
{
  "approved": false,
  "source_snapshot_sha256": null,
  "approved_by": null,
  "approved_at": null
}
```

S2 的独立开放操作才可写 `approved=true`，同时记录准确快照 SHA-256、操作者及该操作发生的带时区时间。批准必须匹配文档的源快照；源快照变化后不能自动沿用旧批准。导入成功、已有 oracle、Public 或 Status=1 均不构成开放操作。S0 不生成批准记录、不开放文档。

S2 实际开放在 `2026-09-29T07:57:58.668281+00:00` 执行，操作者 `codex-local:S2`，范围为上述准确快照的130篇记录。`reports/S2_RELEASES.json` 保存逐 key 结果，`.local/s2/releases.jsonl` 保存操作意图与结果；所有文档仍为 `publication_state=unverified`。开放是普通 Bulk 的逐项更新，不提供批次原子事务，部分失败会明确记录并停止。

```text
有效会话及合法 Principal
AND lab_release.approved 且批准记录完整、源快照匹配
AND acl.state == resolved
AND acl.menu_active_raw == 1
AND (
    acl.is_public
    OR (
        当前唯一角色在 allowed_role_ids
        AND role_status_raw[该角色] == 1
        AND required_business_category 在 effective_business_categories
    )
)
```

对当前全部合法文档统一实验开放后，ACL oracle 的允许数应为：MSFT/21V/双业务 Partner 130、Partner-NCE 102、Partner-OSPA 104。开放前不应把用户读取结果与这组数量直接等同。DLS 必须独立证明整套条件；首次 ES 验证不叠加重复应用 ACL 来掩盖 DLS 缺陷。

## 6. Principal 与会话契约

可信服务端 fixture → `validate_mock_profile()` → `normalize_authentication_info()` → `Principal` 是唯一现有规范化入口。沿用 `Principal.to_dict()`，服务端另外绑定 `profile_id`：

| 字段 | 约束 |
|---|---|
| `profile_id` | 20 个预设 profile 之一，由服务端查表；不是客户端提供的授权集合 |
| `user_id`、`user_name` | 非空字符串，保留独立用户身份 |
| `is_partner` | 严格 bool，与唯一角色是否为 2 一致 |
| `role_ids`、`role_names` | 恰好一项，ID 与名称精确匹配 MSFT=1、Partner=2、21V=3；重复项也拒绝 |
| `raw_business_categories` | 原 BusinessCategories；仅严格整数 0/1，未知编号拒绝 |
| `effective_business_categories` | 内部用户规范化为 `[0,1]`；Partner 将显式有效列表排序去重，不能为空 |
| `business_permission_source` | `internal_default` 或 `partner_explicit` |
| `authz_mode`、`policy_version` | `mock`、`mock-auth-v2-exclusive-role` |

Mock profile 必须 authenticated=true、enabled=true、expired=false；仅显式开启的 development/test 可用，production 始终拒绝。抽取器的无角色/单业务内部角色探针不能作为合法 Principal。外部 key0=OSPA/key1=NCEI 不直接进入本地枚举。

S3 为每个 profile 创建对应 native user，只授予同一个受限 DLS reader。Metadata 由规范化身份产生：`playbook_role_ids`、`playbook_business_categories`、`authz_mode`、`authz_policy_version`，与现有候选文件一致。业务角色不等于 ES 安全角色，不赋予额外管理员业务 profile。

### 6.1 S3 实际身份映射

`tools/es_permissions.py` 已实现 `principals()`、`request_as()` 及显式 `init/check`。先通过原Mock开发/测试环境开关和规范化器校验全部20个profile，再将 `profile_id == Principal.user_name` 作为native用户名；不接受空/混合/重复角色，不做静默去重。profile集合与独立fixture候选metadata逐项一致。

所有业务native用户只有 `pb_document_reader`，精确限定 `playbook-documents-v1` 和独立合成索引 `playbook-s3-policy-checks`，两者各带完整DLS、各自固定快照。没有cluster、run_as、写入或无限制reader；内部角色的双业务来自规范化，不是菜单授权豁免。S3真实/合成逐ID矩阵都直接使用各用户密码，无应用层ACL重过滤。

业务密码只在 `.local/s3/<profile_id>_password`，目录700、文件600；请求读取前校验profile和文件权限，缺失、非法或未知映射失败，不退回管理员。`tools/es_local.request()` 增加仅限调用方内部传递的 `password_value`，每次创建新的Authorization；浏览器从来不提供这个参数。S3 `init` 才使用管理员建立角色/用户，`pb_s3_importer` 仅维护合成索引，真实内容维护仍由S2身份负责。

原fixture和oracle（包括 `es_metadata_candidates.json` 的 `provisioned=false`）保留其原包历史含义；实际部署/检查状态在 `reports/S3_INIT.json`、`S3_CHECKS.json`。合成记录的来源、指纹和固定时间均属于测试数据，不是SQL内容或真实开放记录。S4只查询真实索引，不将合成数据混入业务结果。

### 6.2 S4 会话边界（已实现）

S4 使用同源服务端会话和不透明 HttpOnly cookie，浏览器只提交 profile_id，不传 ES 密码、角色列表、业务授权或 DSL。会话建立/切换后分配新的不透明 `session_revision`，注销/切换使旧会话失效；前端结合本地身份代次与请求序号拒绝迟到响应。多个标签页共享 cookie 时，以服务端当前会话为准，旧 profile 展示不能成为授权依据。

`session_revision` 仅用于响应归属，不能用它授予权限。会话相关响应使用 `Cache-Control: no-store`；首版不加搜索/正文业务缓存。会话失效和撤权不能仅依赖 ES refresh；S6 验证撤权确认后的新请求及旧响应处理。会话写操作只接受同源 JSON 请求并检查来源；mock 不是生产登录。

实际实现为 `api.py`：只绑定127.0.0.1，默认端口18080；cookie 名为 `playbook_session`，HttpOnly、SameSite=Strict、Path=/，固定1小时失效。只在该本机HTTP实验使用，无生产HTTPS登录入口。内存会话上限1000，重启全失效；过期/未知cookie应先DELETE清理再选择用户。切换先使旧token失效，再验证新身份；新身份失败时也不恢复旧会话。DELETE无请求体、成功204；POST要求JSON。所有写请求及搜索POST都验证Origin精确匹配Host，并拒绝跨站来源。

`resolve_identity()` 复用S3的规范化与私有凭据；每次读取核对 `_authenticate` 的native用户名、唯一reader、可信metadata，以及 `_privileges` 中两条精确索引的完整DLS模板、无额外权限。随后只查询真实索引。原Principal发生变化时旧会话拒绝；上游状态异常绝不改用管理员。读取返回前再次检查session_revision，读取过程中被切换/注销的请求返回401。已发出的响应无法从网络撤回，S5仍需取消与版本保护。

## 7. 搜索与读取 API 契约

以下路由已在 S4 实现并完成真实 HTTP 验证。首版 `retrieval_mode` 固定为 `bm25`，不接受客户端选择 provider、任意索引或 Elasticsearch DSL。

| 路由 | 输入与输出边界 |
|---|---|
| `GET /api/dev/profiles` | 仅开发模式；只返回 profile_id 和选择标签，不返回全量权限/期望矩阵 |
| `POST /api/dev/session` | JSON 仅 `{profile_id}`；建立/切换会话，返回 `{request_id,session_revision,principal}`；旧会话失效 |
| `DELETE /api/session` | 注销并清 cookie，返回 204；前端清空旧内容 |
| `GET /api/me` | 当前合法 Principal 及 session_revision，不接收授权覆盖参数 |
| `GET /api/menu` | 授权后的菜单树与可见文档计数；节点含 menu_id、parent_id、label、label_en、type、document_keys、visible_document_count、placeholder、business_category、is_public。无权父节点只用“上级目录”占位；不下发其真实名称、URL 或无权子项 |
| `POST /api/search` | 下述 SearchRequest → SearchResponse，Top K 按 document_key 去重 |
| `GET /api/documents/{key}` | 当前身份授权后返回该文档的标题、双语阅读内容、允许的 metadata、source、版本及 session_revision；不返回内部 ACL 证据或其他文档 |
| `GET /api/documents/{key}/raw?language=zh` | language 仅 zh/en，默认 zh；返回相应字段的原始 UTF-8 文本，不回填另一语言 |
| `GET /api/documents/{key}/download?kind=markdown&language=zh` | kind 仅 markdown/metadata；markdown 的 language 同上。下载前重新授权；metadata 仅含当前可读文档的展示字段，不含内部路径、角色清单或测试期望 |

Raw 与 download 也返回当前会话归属标识（`X-Session-Revision`）及 request ID，不公开磁盘路径。原始空字符串保持空；null 字段返回 204 并标记 `X-Source-Field-Null: true`。下载使用 attachment；原始 HTML 内容不得作为可执行页面返回。key 只解析为固定来源命名空间下的 Document ID，不能拼接成任意路径或切换其他索引。

带会话的成功 JSON 响应均包含 `request_id` 和 `session_revision`：me/session 使用 `principal`，menu 使用 `nodes`，详情使用 `document`，search 使用下述 `items`。Raw/download 的请求标识头为 `X-Request-ID`。前端不得把旧 session_revision 的结果恢复到当前身份视图。

S4菜单返回扁平nodes，按源Sort/Id排序，用parent_id连接。文档keys与可见数量来自当前身份的DLS搜索；源MenuItem/Role_MenuItem/Role只判断导航标签是否可展示，不过滤或扩大DLS文档集合。无权祖先label/label_en均为“上级目录”、type=null；没有URL字段，不含无权兄弟节点。合法的Type2/3导航项可保留，但不引入Notice正文或未开放的文档。

S5增加的business_category/is_public仅为已允许展示的菜单属性，来自同一冻结源关系，用于分组与公开/受控计数；无权祖先这两个字段均为null。页面只按返回的document_keys去重计数，不计算客户端ACL。

详情document含搜索展示字段、metadata白名单（Author/Status/TextFormatType/CreatedTime/UpdatedTime）、publication_state、timestamp_timezone、same_content、双语languages。每个语言返回source_field、html、headings、assets、raw_utf8_sha256、table_warnings、omitted_decorations及raw_is_null/raw_is_empty；原文通过raw单独读取。metadata下载返回同一展示元数据加请求/会话标识，不包含languages或内部ACL。所有raw/Markdown附件均为text/plain、nosniff并有禁止脚本的CSP；不将原始HTML作为页面执行。

### 7.1 SearchRequest

```json
{
  "query": "合同是如何生成的？",
  "top_k": 5,
  "language": "both",
  "business_scope": "all"
}
```

- query：必填字符串，去掉首尾空白后 1–1000 个 Unicode 字符；不是 ES 查询语法。空输入不变成 match_all。
- top_k：严格整数 1–20，默认 5；拒绝 bool、字符串数字和越界值，不静默截断。
- language：`zh`/`en`/`both`，默认 both，用于选择源字段；不是语言检测结论，也不改变 Document ACL。
- business_scope：`all`/整数 0/整数 1，默认 all；只缩小搜索范围，不能扩大授权，不能当成用户业务权限。
- 拒绝未知输入字段，尤其是 role_ids、business_categories、DSL、index、provider、ES 凭据。每次只检索本次 query，不自动拼接历史问题。

### 7.2 SearchResponse

外层：`request_id`（不透明字符串）、`session_revision`、`retrieval_mode="bm25"`、`took_ms`（非负耗时）、`items`（最多 top_k 个不同 Document）。无结果返回 200 和空 items，不能跨权限补足 K。

每个 item 必含：`document_key`、`document_id`、`title`、`title_en`、`business_category`、`is_public`、`matched_language`（zh/en/both）、`score`、`title_highlights`、`snippets`、`source`、`content_revision`、`acl_revision`。

`score` 为 ES 返回的有限非负 BM25 `_score`，保持原数值。结果按 score 降序、同分按 document_id 升序；页面显示两位小数，排序仍用原分数。它只用于同一次查询和授权范围内的相对比较，不归一化为百分比，不跨问题比较，不视为答案正确概率或无答案阈值。

snippet 为 `{source_field, text, parts}`，text 是正文证据的纯文本摘录，不是生成回答；source_field 为 Content/ContentEn。parts 是 `{text, highlighted}` 数组，拼接后必须等于 snippet.text，最多保留300个Unicode字符。title_highlights 仅含 title/title_en 的同形parts，拼接必须还原对应完整标题。matched_language 表示命中的源字段，ContentEn 中的中文仍属于 en 源字段。

高亮使用ES分词/已批准同义词的实际匹配。请求启用`encoder=html`，ES先转义源文本再插入固定mark标签；后端拆为纯文本及布尔标记，前端再次转义文字且只自行生成`<mark>`。源字段中的HTML、实体和看似高亮标签的文本不会被当作页面标记执行。高亮缺失或不能完整还原展示文字时，前端显示转义后的原文字。配置依据见[Elasticsearch高亮设置](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/highlighting-settings)。

对外 source 仅含 `kb_id`、`instance_key`、`table`、`primary_key`、`snapshot_sha256`；不暴露本机路径、密码或不相关文档。不返回全库统计，也不把 `_score` 显示为可信度百分比。标题/章节/正文权重和同义词在后续阶段用实际 token 与问题集验证，不预先宣称效果。

实际查询为每个源语言的multi_match(best_fields)，标题/章节/正文初始权重3/2/1；语言间dis_max(tie_breaker=0)。专用术语通过既有已审核synonym_graph处理，不新增“取消/删除/暂停”同义关系。matched_language由命名语言子查询给出，不由用户问题语言猜测。摘录取该源字段search_text的片段，保留纯文本和独立高亮标记；标题命中而正文未命中时允许取正文开头作为未高亮上下文。空原字段可无摘录，不伪造文字。API实测逐段与S2源文本核对。

没有可切换provider；`bm25_search()` 是内部检索入口。搜索请求关闭ES request cache、拒绝分片部分成功与timed_out；不完整响应不是空结果。ES查询timeout=5s，HTTPS传输每次调用上限30s，took_ms为ES查询耗时，审计elapsed_ms记录包含身份核验的服务端耗时。性能分位数验收仍属S6。

### 7.3 统一错误边界

JSON 错误格式：`{request_id,error:{code,message}}`，不包含凭据、ES 内部请求或堆栈。无权/不存在文档使用相同外部信息。

| HTTP | 含义 |
|---|---|
| 400 | 参数、类型、未知字段或不合法 key；未知 profile 统一视为无效选择 |
| 401 | 会话缺失、失效或身份映射不合法 |
| 403 | 会话写操作的来源检查失败 |
| 404 | 文档不存在或无权读取；开发入口在禁用时也不可用 |
| 503 | ES 或 DLS 能力不可用，包括许可失效；不降级、不改用管理员、不回退本地全库 |
| 504 | 下游检索超时，允许重试；不当作空结果 |

浏览器不直连 ES。后端按当前会话绑定只读 ES 身份；所有读取入口复用同一授权边界，不使用管理员读出后过滤。凭据仅服务端保存；日志仅记录 request ID、身份标识、结果 ID、耗时及必要版本，默认不记录全文、密码或 token。

### 7.4 S5 应用壳与浏览器状态

`build.py`只从`src/`复制三个资源到`web/`。API仅允许GET/HEAD读取`/`、`/index.html`、`/style.css`、`/app.js`，拒绝目录/文件符号链接、路径穿越和其他文件。应用壳使用同源脚本/连接的CSP、no-store、nosniff；原文/下载继续使用S4的禁止脚本CSP。旧全量HTML、portal_data、SQL、fixture、ZIP、原文目录和日志不在静态白名单中。

匿名首页只发现profile标签和当前会话；合法会话再读取菜单，正文只在打开具体Document或深链时加载。ASK AI只保留最新一次提交的结果，标题/摘录作为文本转义，不使用本地全文搜索、LLM、多轮上下文或持久化正文缓存。

ASK AI使用原生模态弹窗，桌面入口位于header中央，1000px及以下使用右侧悬浮入口。点击或`⌘ / Ctrl + K`打开并聚焦问题；Esc、关闭按钮或遮罩关闭，焦点返回打开前的位置。点击结果立即关闭弹窗，将焦点交给阅读区，后端按当前会话读取文档并更新URL和目录选中项。普通关闭保留当前会话最近的问题/结果；会话清理会关闭弹窗并清空其内容，核对会话期间弹窗正文同样隐藏。

切换/注销同步清空菜单、正文、原文、弹窗、问题、结果和下载临时URL。每类请求使用独立AbortController和请求票据；票据绑定本地身份代次，读取完整响应后再验证当前票据和服务端session_revision。旧请求的401也不能清掉新会话。详情切换语言/原文另有视图序号，避免同文档内的迟到更新。

BroadcastChannel只广播“变更中/已变更”，不传Principal或凭据；其他标签页清空后通过`/api/me`重新确认服务端cookie身份。焦点/可见性恢复时隐藏旧内容并核对会话，pagehide清空内容，恢复页面重新确认会话。URL只包含Document ID与语言，后退、刷新和复制链接仍重新授权。下载按钮先请求授权接口并校验响应归属，再短暂创建单篇Blob URL，切换时取消并撤销；不是离线全库下载载荷。

## 8. 评测与后续阶段边界

2026-09-29 用户已审核原32条V1验证问题，随后明确回复“确认草案与 90% 目标”。S6冻结 `evaluation/golden_v1.json`，含5类代表身份的可访问参考、相关性分级和3条无答案样例。原 `candidate_questions.json` 保留问题批准时的历史状态和全文指纹；当前金标状态以独立golden文件为准，不改写原问题或参考ID。真实业务提交仍未提供；当前三组同义词继续使用，新增须审核并版本化。

2,600条授权断言只检验ACL；不可访问参考不能计为相关性漏召回。V1已冻结每类身份有可访问参考的核心问题Hit@5≥90%。等级2表示直接相关，等级1表示相关外链/背景；Q23只有外链，记等级1。未标注文档为未知；Recall@5只指已标注参考集合，不能称全库召回率。评测前检查批准指纹、原问题指纹、源字节和实验开放状态。无答案返回率单列，不使用未经标定的BM25阈值；界面提醒查阅原文确认答案。

S6实测报告为 `reports/S6_EVALUATION.json`；32×5组合单列结果及排除原因，暖机25次、测量480次、单并发，另测3×5无答案组合。ES took和HTTP端到端墙钟分别记录p50/p95、硬件和缓存条件，数值不构成生产性能保证。更新/撤权实测为 `reports/S6_CHANGES.json`；离线、ES、API、浏览器证据分别记录。

## 9. S6 手工完整快照与同步窗口

`tools/sync_playbook.py`只在显式development/test mock环境执行。`export --complete`静态解析原SQL的六表，`complete:true`是操作者对完整性的明确声明；程序还检查表集合、主键、关系、菜单路径、互斥角色、状态、身份集合及ACL可解析性。完整性声明不能替代上游导出的一致性保证。拒绝不完整包、歧义关系、未知状态和新增任意身份。当前上限1,000篇、10 MiB源包及bulk，整批处理；无CDC或定时同步。

受控包使用`schema=s6-full-snapshot-v1`，包含Document、MenuItem、MenuItem_Document、Role、Role_MenuItem、Notice、可信profile集合和`withdrawn_document_ids`。SQL字节hash保留在`origin.sql_sha256`；新批次的`source.snapshot_sha256`为六张解析表的canonical hash，不能冒称SQL字节hash。原SQL、fixtures和oracle不覆盖；每个受控包、计划和开放决定保存在700目录、600文件的`.local/s6/local/`。合成验证固定用`.local/s6/validation/`及专用索引/账号，不混入业务资料。

`plan`先从正式关系重建记录，报告新增、删除、正文、ACL、metadata、导航和profile差异；不依赖Document.UpdatedTime。计划以内容hash命名，绑定当前generation及实际索引指纹。`apply`重新校验，持有写锁后拒绝过期计划或索引漂移；删除必须另带`--allow-delete`。新源快照默认关闭，只有明确`--approve-release --approved-by ...`才开放；同源重复导入保留既有开放记录并且零写入。withdrawn列表和inactive菜单始终关闭。

当前选用**单机维护窗口内更新固定索引**：持久化maintenance标记阻止新API请求，等已开始的读取完成发送；收回reader索引权限（真实环境同时收回S2 probe）；写入完整批次、等待refresh、逐条比对整个索引；更新native用户metadata/启用状态；恢复只含当前快照的完整DLS并逐身份核对；原子发布菜单/profile/generation后删除维护标记并确认成功。读取路径只有各自用户凭据；管理员只改角色/用户，不代查业务文档。该方案允许短暂停读，无别名或跨主机分布式事务。

中途失败保留持久化门禁并撤回reader权限；API返回503，不能回退本地全库。进程异常退出后锁自动释放，但门禁仍在；只能以**同一计划和原开放/删除决定**重试。成功发布前不更新active；失败后不自动恢复旧ACL。若重试所需源包/凭据丢失，停止服务并人工恢复，不删除门禁强行放行。S6接管后旧S2导入/开放和S3初始化入口拒绝写入，防止重新引入旧权限；读取验收仍可运行，但冻结oracle/金标只适用于原快照。

上游变化至本地确认的窗口为**人工导出、审核和执行所需时间＋实测维护时间**，无自动实时同步承诺。apply返回成功后新API请求必须使用新generation，旧cookie返回401；新native查询按新DLS/metadata处理。API no-store，ES request cache关闭，前端继续用身份代次/请求序号和session_revision拒绝迟到结果。可见页面每15秒核对一次会话，窗口恢复焦点/可见时也核对；成功确认401或故障后立即清空正文、目录、搜索、弹窗及下载临时URL。后台节流/网络延迟不包含在15秒定时周期承诺内；已发送给用户的字节和已保存下载无法撤回。

`rollback-content --from-batch ...`只向最新包中仍存在的Document复制Content/ContentEn；保留最新ACL、菜单、metadata、profile、撤回和删除集合。回滚包绑定生成时的policy_batch；审核期间发生新更新则拒绝plan，必须从最新批次重新生成；计划形成后再变化也会在apply拒绝过期generation。回滚包需要重新plan/apply及显式开放，已应用的旧回滚包不能直接重复用作新权限变更包。旧物理索引不在reader的精确授权索引列表中，历史原文/计划文件不在HTTP白名单内，raw/download继续逐请求授权。合成旧索引的search/GET返回403；mget的HTTP 200中每项为security_exception，无_source，不把外层200误判为可读。

更新既有native用户时不传password以保留其凭据，遵循[Elasticsearch用户更新接口](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-security-put-user)。运行时仍核对当前native身份、metadata及完整DLS，不依赖本地菜单过滤修补权限。
