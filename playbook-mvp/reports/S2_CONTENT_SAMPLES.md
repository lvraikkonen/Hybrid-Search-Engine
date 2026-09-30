# S2 原文、检索文本与表格抽样

源快照：`cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821`。原文在本机忽略目录逐字节保留；此处只展示检索文本抽样。

## Document 25：合同的生成

业务 0；正式菜单 31；SQL 行 [2985, 3005]；中英源字段相同：False。

```text
通过审批的信息表将提交至微软CE用以生成合同文件。
相关方：合作伙伴客户经理、客户、微软合作伙伴经理、微软CE、21Vianet合约管理部、法务部。
相关系统：
```

表格警告：`[]`。

## Document 63：21VCA-E Playbook访问权限

业务 0；正式菜单 77；SQL 行 [5474, 6442]；中英源字段相同：False。

```text
Catalogue[1]：1 | Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E | Title[3]：1.1 | Title[4]：21VCA-E新商务模式介绍 Introduction to 21VCA-E New Commerce | Customer[5]：✔️ | Partner[6]：✔️ | Sales[7]：✔️
Catalogue[1]：1 | Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E | Title[3]：1.2 | Title[4]：21VCA、OSPA渠道区别 21VCA/OSPA Differences in Channels | Customer[5]：〔空白〕 | Partner[6]：✔️ | Sales[7]：✔️
Catalogue[1]：1 | Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E | Title[3]：1.3 | Title[4]：由世纪互联运营的Microsoft Azure产品 Microsoft Azure Operated by 21Viane | Customer[5]：✔️ | Partner[6]：✔️ | Sales[7]：✔️
```

表格警告：`[]`。

## Document 99：合同的生成

业务 1；正式菜单 114；SQL 行 [9689, 9697]；中英源字段相同：False。

```text
Direct Model: 客户确认购买产品后，微软销售联系微软CE生成合同文件。
Indirect Model: 客户确认购买产品后，合作伙伴/微软CE出合同，生成合同文件。
相关系统：Hermes、E-Sign世纪互联蓝云电子签系统。
```

表格警告：`[]`。

## Document 148：OSPA Playbook访问权限

业务 1；正式菜单 174；SQL 行 [12246, 13234]；中英源字段相同：True。

```text
Catalogue[1]：5 | Catalogue[2]：如何签署世纪互联在线服务高级协议 The Customer Journey of OSPA | Title[3]：5.5 | Title[4]：合同的生成 Contract Creation | Customer[5]：〔空白〕 | Partner[6]：✔️ | Sales[7]：✔️
```

表格警告：`["row_58:rowspan_8_exceeds_7_existing_group_rows", "row_58:rowspan_8_exceeds_7_existing_group_rows"]`。

合并表格按正式单元格展开：列标题、目录项及行值保持关联。空单元格显示为〔空白〕，不推导为允许/拒绝。Document 148 的末段 rowspan=8 超出实际 7 行，仅重复现有行的标签；原始 HTML 的 rowspan 保持不变。

Document 148 正文虽是权限说明表，应用 ACL 仍只来自 SQL 正式菜单关系；不从这些表格内容生成授权。英文源字段混有中文时仍作为 ContentEn 保留；相同语言字段不删除或翻译。
