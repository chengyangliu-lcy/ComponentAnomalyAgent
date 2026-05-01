from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4
import unittest

from schemas import Evidence
from tools.circuit_kb import (
    CircuitMarkdownRetriever,
    build_filtered_public_kb,
    build_circuit_md_kb,
    build_fts_query,
    classify_query_terms,
    chunk_text,
    clean_public_tech_text,
    expand_chinese_electronics_terms,
    is_public_tech_kb_page,
    is_boilerplate_text,
    is_useful_page,
    parse_markdown_page,
    parse_markdown_metadata,
)


class CircuitKbTests(unittest.TestCase):
    def test_parse_markdown_page_extracts_metadata_and_body(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_12.md"
            path.write_text(
                """
**标题**: MOSFET Current Sense Debug

**链接**: https://example.com/mosfet-current-sense

**发布时间**: 2024-01-02


## Current sense notes

R5 is a 0.1 ohm shunt and the LM358 filters leading edge noise.
""".strip(),
                encoding="utf-8",
            )

            page = parse_markdown_page(path)

        self.assertEqual(page.title, "MOSFET Current Sense Debug")
        self.assertEqual(page.url, "https://example.com/mosfet-current-sense")
        self.assertEqual(page.published_at, "2024-01-02")
        self.assertIn("LM358", page.text)
        self.assertNotIn("**标题**", page.text)

    def test_noise_filter_rejects_object_moved_redirect(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_18.md"
            path.write_text(
                """
**标题**: Object moved

**链接**: http://circuitmaker.com/Stream

**发布时间**: 未知

## Object moved to here.
""".strip(),
                encoding="utf-8",
            )

            page = parse_markdown_page(path)

        self.assertFalse(is_useful_page(page))

    def test_noise_filter_rejects_server_error_pages(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_21.md"
            path.write_text(
                """
**标题**: The resource cannot be found.

**链接**: https://circuitmaker.com/Projects/Details/example/CON

**发布时间**: 未知

# Server Error in '/' Application.
## The resource cannot be found.
Description: HTTP 404. The resource you are looking for might have been removed, had its name changed, or is temporarily unavailable.
""".strip(),
                encoding="utf-8",
            )

            page = parse_markdown_page(path)

        self.assertFalse(is_useful_page(page))

    def test_noise_filter_rejects_page_template_boilerplate(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_19.md"
            path.write_text(
                """
**标题**: LED Driver layout test

**链接**: https://circuitmaker.com/Projects/Details/example

**发布时间**: 未知

Add The Following Snippet To Your Html
<iframe frameborder="0" src="https://example.com"></iframe>
Please wait... Add New Maker Fabricate Download Files Delete Components Files
""".strip(),
                encoding="utf-8",
            )

            page = parse_markdown_page(path)

        self.assertFalse(is_useful_page(page))

    def test_noise_filter_rejects_circuitmaker_navigation_and_legal_text(self) -> None:
        text = (
            "/Images/ajax-loader.gif) * Home * Projects * Hubs * Components * Forum * Makers * About "
            "Copyright © 2021 Altium Limited Privacy Policy Cookie Policy GNU General Public License "
            "Do not show this message again Add New Maker Download Files Delete Components Files "
            "LED driver MOSFET current."
        )

        self.assertTrue(is_boilerplate_text(text))

    def test_noise_filter_rejects_circuitmaker_user_pages(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_20.md"
            _write_page(
                path,
                "Chris Short",
                "https://circuitmaker.com/User/Details/Chris-Short",
                "MOSFET LED buck current sense project list and community profile text. " * 8,
            )

            page = parse_markdown_page(path)

        self.assertFalse(is_useful_page(page))

    def test_public_tech_filter_keeps_articles_and_rejects_low_value_sources(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "串行AD和DA芯片的应用 - 布线技巧与EMC - 电子发烧友网",
                "http://www.elecfans.com/article/80/114/2009/2009061669056.html",
                "串行AD和DA芯片布线时需要注意模拟地和数字地、电源退耦、电流回路和EMC干扰。 " * 10,
            )
            _write_page(
                source / "page_2.md",
                "consumer electronics Archives | Electronics For You",
                "https://www.electronicsforu.com/tag/consumer-electronics",
                "Home Projects News Tag Category sign in archive listing. " * 10,
            )
            _write_page(
                source / "page_3.md",
                "Reflow Oven Controller | Projects | CircuitMaker",
                "https://circuitmaker.com/Projects/Details/example/reflow",
                "Download Files Add New Maker Fabricate Please Wait. " * 10,
            )
            pages = [parse_markdown_page(path) for path in sorted(source.glob("page_*.md"))]

            meta = build_filtered_public_kb([source], output, max_docs=10, min_chunk_chars=120)
            results = CircuitMarkdownRetriever(output).search("AD DA EMC 布线 模拟地 数字地", limit=2)

        self.assertTrue(is_public_tech_kb_page(pages[0]))
        self.assertFalse(is_public_tech_kb_page(pages[1]))
        self.assertFalse(is_public_tech_kb_page(pages[2]))
        self.assertEqual(meta["documents"], 1)
        self.assertTrue(results)
        self.assertIn("elecfans.com/article", results[0].source)

    def test_elecfans_cleaner_keeps_article_body_and_drops_recommendation_blocks(self) -> None:
        raw = """
  * 网站导航
  * 首页
  * 技术资料
电子搜索 搜索技术文章、芯片资料等 请输入元器件型号
#### 编辑推荐
##### 联发科、高通、NXP等众多半导体厂商快充方案全集
电子技术应用频道 -- 为电子工程师提供电子产品设计所需的技术分析、设计技巧、设计工具、测试工具等技术文章！
#### 推荐帖子
##### 【社区之星】FPGA开发工程师浅谈学习FPGA的正确打开方式
# 运放恒流源电路原理
_2016年06月01日 10:00_ 来源：电子发烧友
运放恒流源不是开环比较器，而是通过负反馈调节MOSFET栅极。
采样电阻Rsense上的电压等于基准电压，输出电流公式为I=Vref/Rsense。
当MOSFET发热或输出振荡时，应检查补偿电容、采样走线和栅极电阻。
## 相关技术文章：
  * 三星豪赌5G移动网络
## 相关资料下载：
  * 手机电路原理及检修
### 上周热点文章排行榜
  1. 总结：关于直流电防接反电路
## 用户评论
发表评论
""".strip()

        cleaned = clean_public_tech_text(raw)

        self.assertIn("运放恒流源不是开环比较器", cleaned)
        self.assertIn("I=Vref/Rsense", cleaned)
        self.assertIn("补偿电容", cleaned)
        self.assertNotIn("网站导航", cleaned)
        self.assertNotIn("编辑推荐", cleaned)
        self.assertNotIn("推荐帖子", cleaned)
        self.assertNotIn("电子技术应用频道", cleaned)
        self.assertNotIn("相关技术文章", cleaned)
        self.assertNotIn("上周热点文章排行榜", cleaned)
        self.assertNotIn("用户评论", cleaned)

    def test_public_tech_filter_rejects_elecfans_404_title(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_404.md"
            _write_page(
                path,
                "404 页面不存在！ - 中国电子工程师最喜欢的电子发烧友网",
                "http://www.elecfans.com/analog/20180129625516.html",
                "电子发烧友网 技术文章 电源技术 模拟技术 推荐内容 " * 20,
            )

            page = parse_markdown_page(path)

        self.assertFalse(is_public_tech_kb_page(page))

    def test_public_tech_filter_rejects_elecfans_category_and_study_download_pages(self) -> None:
        with _workspace_tempdir() as root:
            category = Path(root) / "page_category.md"
            study = Path(root) / "page_study.md"
            _write_page(
                category,
                "稳压电源_电源/新能源-电子发烧友网",
                "http://www.elecfans.com/article/83/144/",
                "USB Type-C 反激式转换器 MOSFET 电感器 稳压器 分类列表 " * 20,
            )
            _write_page(
                study,
                "轻型空箱围堤对邻近桥梁桩基影响分析-电子电路图,电子技术资料网站",
                "http://www.elecfans.com/soft/study/build/2010/2010042473930.html",
                "围堤 桥梁 桩基 负摩阻力 侧向推力 数值模拟 " * 20,
            )

            category_page = parse_markdown_page(category)
            study_page = parse_markdown_page(study)

        self.assertFalse(is_public_tech_kb_page(category_page))
        self.assertFalse(is_public_tech_kb_page(study_page))

    def test_elecfans_cleaner_keeps_soft_intro_body_and_drops_download_navigation(self) -> None:
        raw = """
| ![](/skin/sky-cn_net/EET_Sch_01.gif) | 文章:新闻┆EDA技术┆电源技术┆无线通信 |
下载:EDA教程┆电源技术┆电子书籍┆电子元件┆无线通信┆通信网络
栏目导航
热门下载
技术资料介绍
![](/images/load.gif)
SN3910是一款Universal High Brightness LED Driver，可用于高压LED恒流驱动。
芯片通过电感电流控制LED电流，支持PWM调光和过温保护。
相关下载
""".strip()

        cleaned = clean_public_tech_text(raw)

        self.assertIn("SN3910是一款", cleaned)
        self.assertIn("PWM调光", cleaned)
        self.assertNotIn("文章:新闻", cleaned)
        self.assertNotIn("下载:EDA教程", cleaned)
        self.assertNotIn("栏目导航", cleaned)

    def test_filtered_public_builder_skips_oversized_files_before_parsing(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "运放恒流源电路原理 - 电子发烧友网",
                "http://www.elecfans.com/article/analog/2009/2009061669056.html",
                "运放恒流源通过负反馈调节MOSFET栅极，I=Vref/Rsense。 " * 12,
            )
            _write_page(
                source / "page_2.md",
                "超大导航页 - 电子发烧友网",
                "http://www.elecfans.com/article/noisy/huge.html",
                "网站导航 推荐帖子 相关推荐 " * 5000,
            )

            meta = build_filtered_public_kb([source], output, max_docs=10, max_page_num=2, max_file_bytes=3000, min_chunk_chars=120)

        self.assertEqual(meta["documents"], 1)
        self.assertEqual(meta["stats"]["oversized_files"], 1)

    def test_parse_markdown_metadata_reads_header_without_body(self) -> None:
        with _workspace_tempdir() as root:
            path = Path(root) / "page_1.md"
            path.write_text(
                """
**标题**: 运放恒流源电路原理 - 电子发烧友网

**链接**: http://www.elecfans.com/article/analog/2009/2009061669056.html

**发布时间**: 2024-01-02

""" + ("正文很长 " * 1000),
                encoding="utf-8",
            )

            metadata = parse_markdown_metadata(path, max_header_bytes=240)

        self.assertEqual(metadata["title"], "运放恒流源电路原理 - 电子发烧友网")
        self.assertEqual(metadata["url"], "http://www.elecfans.com/article/analog/2009/2009061669056.html")
        self.assertEqual(metadata["published_at"], "2024-01-02")

    def test_build_circuit_md_kb_and_retrieve_expected_chunk(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_2.md",
                "LED Blink",
                "https://example.com/led",
                "Blink an LED with Arduino delay and a GPIO output resistor. " * 8,
            )
            _write_page(
                source / "page_1.md",
                "MOSFET Current Sense Filter",
                "https://example.com/current-sense",
                "The LM358 op amp filters MOSFET current sense noise from R5 shunt measurements. " * 8,
            )
            _write_page(
                source / "page_3.md",
                "Object moved",
                "https://example.com/moved",
                "Object moved to here.",
            )

            meta = build_circuit_md_kb(source, output, max_docs=10, chunk_chars=500, chunk_overlap=50, min_chunk_chars=120)
            retriever = CircuitMarkdownRetriever(output)
            results = retriever.search("LM358 MOSFET current sense R5 noise", limit=2)
            stored_meta = json.loads((output / "build_meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["documents"], 2)
        self.assertEqual(stored_meta["documents"], 2)
        self.assertTrue(results)
        self.assertEqual(results[0].metadata["kind"], "local_kb_chunk")
        self.assertIn("current sense", results[0].content.lower())
        self.assertIn("bm25_score", results[0].metadata)
        self.assertIn("rerank_score", results[0].metadata)

    def test_rerank_prefers_title_and_component_matches(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "Generic Motor Driver",
                "https://example.com/generic",
                "A motor driver article mentions current sense once but mostly discusses software setup. " * 8,
            )
            _write_page(
                source / "page_2.md",
                "LM358 R5 Current Sense Noise",
                "https://example.com/lm358-r5",
                "The op amp input and R5 shunt create a current sense signal for MOSFET protection. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search("LM358 R5 current sense", limit=1)

        self.assertEqual(results[0].source, "https://example.com/lm358-r5")

    def test_chinese_question_expands_to_english_circuit_terms(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "MOSFET Current Sense Filter",
                "https://example.com/current-sense",
                "The LM358 op amp filters MOSFET current sense noise from the R5 shunt resistor. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=500, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search("运放电流采样噪声怎么处理 R5", limit=1)

        self.assertTrue(results)
        self.assertEqual(results[0].source, "https://example.com/current-sense")
        self.assertIn("current sense", results[0].content.lower())

    def test_local_retrieve_executor_uses_circuit_retriever(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "NTC Inrush Limiter",
                "https://example.com/ntc",
                "An NTC limits inrush current before the bridge rectifier and bulk capacitor charge. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, min_chunk_chars=120)

            try:
                from tools.evidence_tools import LocalRetrieveExecutor
            except ModuleNotFoundError as exc:
                self.skipTest(f"optional agent tool dependency is unavailable: {exc}")

            executor = LocalRetrieveExecutor(CircuitMarkdownRetriever(output))
            run = executor.run("NTC inrush current bridge rectifier", limit=1)

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].metadata["kind"], "local_kb_chunk")
        self.assertIn("NTC", run.evidence[0].title)

    def test_local_retrieve_executor_reports_missing_index_as_error(self) -> None:
        with _workspace_tempdir() as root:
            missing = Path(root) / "missing_kb"

            try:
                from tools.evidence_tools import LocalRetrieveExecutor
            except ModuleNotFoundError as exc:
                self.skipTest(f"optional agent tool dependency is unavailable: {exc}")

            executor = LocalRetrieveExecutor(CircuitMarkdownRetriever(missing))
            run = executor.run("MC34063 buck LED constant current driver", limit=1)

        self.assertFalse(run.success)
        self.assertIn("local KB index unavailable", run.errors[0])
        self.assertFalse(run.metadata["usable"])

    def test_model_query_requires_model_match_after_rerank(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "Generic LED Driver layout",
                "https://example.com/led-layout",
                "LED driver layout test with boost ripple and TVS notes but no controller model. " * 8,
            )
            _write_page(
                source / "page_2.md",
                "LTC3896 72V input protection",
                "https://example.com/ltc3896",
                "LTC3896 high voltage input supply can burn without TVS clamp, layout guidelines, and switch node protection. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search("LTC3896 72V input burns layout TVS", limit=2)

        self.assertTrue(results)
        self.assertEqual(results[0].source, "https://example.com/ltc3896")
        self.assertTrue(all("LTC3896" in item.content or "LTC3896" in item.title for item in results))

    def test_model_query_does_not_return_unmatched_model_chunk(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "Generic Buck LED Constant Current Driver",
                "https://example.com/generic-buck",
                "A buck LED constant current driver uses current sense feedback and a Schottky diode. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search("MC34063 buck LED constant current driver", limit=2)

        self.assertEqual(results, [])

    def test_digit_prefixed_models_are_strong_required_terms(self) -> None:
        profile = classify_query_terms("PNP NPN current limit circuit 2SA1416 2PB709BSL schematic")

        self.assertIn("2SA1416", profile["models"])
        self.assertIn("2PB709BSL", profile["models"])
        self.assertNotIn("PNP", profile["models"])
        self.assertNotIn("NPN", profile["models"])

    def test_plp_and_mixed_case_product_terms_are_strong_required_terms(self) -> None:
        profile = classify_query_terms("PLP capacitor selection ActiveCips Qorvo")

        self.assertIn("PLP", profile["models"])
        self.assertIn("ActiveCips", profile["models"])

    def test_voltage_values_and_emc_are_not_overstrong_model_terms(self) -> None:
        profile = classify_query_terms("48V BMS BCU CSU EMC common mode choke 12V Type-C")

        self.assertIn("48V", profile["values"])
        self.assertIn("12V", profile["values"])
        self.assertIn("BMS", profile["models"])
        self.assertIn("BCU", profile["models"])
        self.assertIn("CSU", profile["models"])
        self.assertNotIn("48V", profile["models"])
        self.assertNotIn("EMC", profile["models"])
        self.assertNotIn("Type-C", profile["models"])

    def test_bms_query_rejects_led_driver_chunk_that_only_matches_voltage(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "SN3910 LED Driver",
                "https://example.com/sn3910",
                "SN3910 LED driver feedback dimming for -48V LED current and thermal protection. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search(
                "48V BMS architecture BCU CSU isolation design differences from 12V",
                limit=2,
            )

        self.assertEqual(results, [])

    def test_plp_query_rejects_generic_buck_charger_project_page(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "Buck-Boost Multi-Chemistry Battery Charger with MPPT",
                "https://circuitmaker.com/Projects/Details/Craig-Peacock-4/LT8490-Buck-Boost-Multi-Chemistry-Battery-Charger-with-MPPT",
                "This buck-boost charger project discusses capacitor voltage rating and current selection. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search(
                "PLP circuit capacitor selection capacitance vs voltage rating ActiveCips Qorvo",
                limit=2,
            )

        self.assertEqual(results, [])

    def test_model_query_rejects_circuitmaker_project_without_model_match(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "lab supply2",
                "https://circuitmaker.com/Projects/F48BC6C2-9005-48B8-87CD-C89CD40F70DC",
                "PNP NPN current limit schematic using a relay and opamp comparator. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            results = CircuitMarkdownRetriever(output).search(
                "PNP NPN current limit circuit 2SA1416 2PB709BSL schematic",
                limit=2,
            )

        self.assertEqual(results, [])

    def test_local_retrieve_executor_reports_filtering_diagnostics(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            source = root_path / "source"
            output = root_path / "kb"
            source.mkdir()
            _write_page(
                source / "page_1.md",
                "Buck Current Sense Design",
                "https://example.com/buck-current-sense",
                "Buck converter current sense resistor feedback calculation and ripple measurement. " * 8,
            )
            build_circuit_md_kb(source, output, max_docs=10, chunk_chars=600, chunk_overlap=50, min_chunk_chars=120)

            try:
                from tools.evidence_tools import LocalRetrieveExecutor
            except ModuleNotFoundError as exc:
                self.skipTest(f"optional agent tool dependency is unavailable: {exc}")

            run = LocalRetrieveExecutor(CircuitMarkdownRetriever(output)).run("buck current sense ripple", limit=2)

        self.assertTrue(run.success)
        self.assertGreaterEqual(run.metadata["kb_candidate_count"], 1)
        self.assertEqual(run.metadata["kb_used_count"], len(run.evidence))

    def test_local_retrieve_executor_only_returns_high_relevance_evidence(self) -> None:
        try:
            from tools.evidence_tools import LocalRetrieveExecutor
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")

        class FakeStatus:
            usable = True

            def to_json(self):
                return {"usable": True}

        class FakeRetriever:
            index_dir = Path("fake-index")

            def status(self):
                return FakeStatus()

            def search_with_diagnostics(self, query, limit=4):
                return {
                    "evidence": [
                        Evidence(
                            source="https://example.com/weak",
                            title="weak buck note",
                            content="Buck current sense ripple note.",
                            score=8.0,
                            metadata={"kind": "local_kb_chunk", "kb_relevance": 8.0, "high_relevance": False},
                        )
                    ],
                    "diagnostics": {"candidate_count": 3, "discarded_kb": 2, "high_relevance_count": 0},
                }

        run = LocalRetrieveExecutor(FakeRetriever()).run("buck current sense ripple", limit=2)

        self.assertFalse(run.success)
        self.assertEqual(run.evidence, [])
        self.assertEqual(run.metadata["kb_candidate_count"], 3)
        self.assertEqual(run.metadata["kb_used_count"], 0)
        self.assertEqual(run.metadata["high_relevance_count"], 0)

    def test_evidence_rank_discards_low_relevance_kb(self) -> None:
        try:
            from tools.evidence_tools import EvidenceRankExecutor
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")
        low_quality = {
            "source": "https://circuitmaker.com/Projects/Details/example",
            "title": "CircuitMaker template",
            "content": "Home * Projects * Hubs * Components * Forum Copyright Privacy Policy LED MOSFET",
            "score": 1.0,
            "metadata": {"kind": "local_kb_chunk", "kb_relevance": 1.0},
        }

        run = EvidenceRankExecutor().run("buck current sense ripple", evidence=[_evidence(low_quality)], max_items=3)

        self.assertEqual(run.evidence, [])
        self.assertEqual(run.metadata["discarded_kb"], 1)

    def test_evidence_rank_discards_kb_not_marked_high_relevance(self) -> None:
        try:
            from tools.evidence_tools import EvidenceRankExecutor
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")
        kb_chunk = {
            "source": "https://www.elecfans.com/power/buck-current-sense.html",
            "title": "Buck current sense layout",
            "content": "Buck converter current sense resistor feedback calculation and ripple measurement.",
            "score": 8.0,
            "metadata": {
                "kind": "local_kb_chunk",
                "kb_relevance": 8.0,
                "high_relevance": False,
                "matched_query_terms": {"topology": ["buck"], "fault": ["ripple"]},
            },
        }

        run = EvidenceRankExecutor().run("buck current sense ripple", evidence=[_evidence(kb_chunk)], max_items=3)

        self.assertEqual(run.evidence, [])
        self.assertEqual(run.metadata["discarded_kb"], 1)

    def test_evidence_rank_discards_circuitmaker_project_without_strong_match(self) -> None:
        try:
            from tools.evidence_tools import EvidenceRankExecutor
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")
        project_chunk = {
            "source": "https://circuitmaker.com/Projects/Details/example/Generic-Buck-Charger",
            "title": "Generic Buck Charger",
            "content": "Buck charger capacitor voltage current selection with schematic notes.",
            "score": 6.0,
            "metadata": {"kind": "local_kb_chunk", "kb_relevance": 6.0, "matched_query_terms": {"topology": ["buck"]}},
        }

        run = EvidenceRankExecutor().run(
            "PLP circuit capacitor selection capacitance vs voltage rating ActiveCips Qorvo",
            evidence=[_evidence(project_chunk)],
            max_items=3,
        )

        self.assertEqual(run.evidence, [])
        self.assertEqual(run.metadata["discarded_kb"], 1)

    def test_finish_answer_discards_kb_without_question_term_overlap(self) -> None:
        try:
            from tools.evidence_tools import FinishAnswerExecutor
            from llm_client import LLMClient
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")
        unrelated_kb = _evidence(
            {
                "source": "https://www.elecfans.com/dianlutu/LED/20100225170410.html",
                "title": "强光LED手电筒电路",
                "content": "LED手电筒使用升压电路和限流电阻，介绍开关管和电感选择。",
                "score": 8.0,
                "metadata": {
                    "kind": "local_kb_chunk",
                    "kb_relevance": 8.0,
                    "matched_query_terms": {"topology": ["LED"]},
                },
            }
        )

        run = FinishAnswerExecutor(LLMClient(api_key=None, base_url="", model="test")).run(
            "PLP circuit capacitor selection capacitance vs voltage rating ActiveCips Qorvo",
            evidence=[unrelated_kb],
            allow_llm=False,
        )

        self.assertNotIn("强光LED手电筒", run.text)
        self.assertNotIn("本地知识库", run.text)

    def test_finish_answer_discards_kb_not_marked_high_relevance(self) -> None:
        try:
            from tools.evidence_tools import FinishAnswerExecutor
            from llm_client import LLMClient
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional agent tool dependency is unavailable: {exc}")
        low_certainty_kb = _evidence(
            {
                "source": "https://www.elecfans.com/power/buck-current-sense.html",
                "title": "Buck current sense layout",
                "content": "Buck converter current sense resistor feedback calculation and ripple measurement.",
                "score": 8.0,
                "metadata": {
                    "kind": "local_kb_chunk",
                    "kb_relevance": 8.0,
                    "high_relevance": False,
                    "matched_query_terms": {"topology": ["buck"], "fault": ["ripple"]},
                },
            }
        )

        run = FinishAnswerExecutor(LLMClient(api_key=None, base_url="", model="test")).run(
            "buck current sense ripple",
            evidence=[low_certainty_kb],
            allow_llm=False,
        )

        self.assertNotIn("Buck current sense layout", run.text)
        self.assertNotIn("本地知识库", run.text)

    def test_chunk_text_and_query_helpers(self) -> None:
        self.assertGreaterEqual(len(chunk_text("LM358 current sense. " * 120, chunk_chars=300, overlap=50, min_chars=120)), 3)
        self.assertIn('"LM358"', build_fts_query("LM358 R5 current sense"))
        self.assertIn("current sense", expand_chinese_electronics_terms("电流采样噪声"))

    def test_local_retrieve_repair_expands_chinese_query_to_english(self) -> None:
        from agent.tool_argument_repair import repair_action_args

        repaired, notes = repair_action_args(
            "local_retrieve",
            {"query": "运放电流采样噪声怎么处理 R5", "limit": 10},
            question="LM358 运放电流采样噪声怎么处理 R5",
            evidence=[],
            max_web_results=5,
            rank_limit=4,
            next_seed_query=lambda: "",
            select_read_target=lambda _url: None,
            allow_llm=True,
        )

        self.assertEqual(repaired["limit"], 4)
        self.assertIn("op amp", repaired["query"])
        self.assertIn("current sense", repaired["query"])
        self.assertIn("R5", repaired["query"])
        self.assertTrue(notes.applied)


def _write_page(path: Path, title: str, url: str, body: str) -> None:
    path.write_text(
        f"""
**标题**: {title}

**链接**: {url}

**发布时间**: 未知

## {title}

{body}
""".strip(),
        encoding="utf-8",
    )


def _evidence(payload: dict) -> Evidence:
    return Evidence(**payload)


class _workspace_tempdir:
    def __enter__(self) -> str:
        base = Path(__file__).resolve().parents[1] / "outputs" / "test_circuit_kb"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / uuid4().hex
        self.path.mkdir(parents=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
