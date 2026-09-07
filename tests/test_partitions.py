from rag.partitions import (
    DEFAULT_PARTITION_CONFIG,
    PartitionRegistry,
    load_partition_registry,
)


def _registry() -> PartitionRegistry:
    registry = load_partition_registry(DEFAULT_PARTITION_CONFIG)
    assert registry is not None
    return registry


def test_registry_loads_partitions_and_collection() -> None:
    registry = _registry()
    assert registry.collection_name == "medical_knowledge_base"
    assert registry.default_partition == "diagnosis"
    assert registry.partition_ids == ("device", "medication", "diagnosis")
    assert set(registry.partitions) == {"diagnosis", "device", "medication"}


def test_load_partition_registry_is_none_when_disabled_or_missing() -> None:
    assert load_partition_registry(None) is None
    missing = DEFAULT_PARTITION_CONFIG.parent / "does_not_exist.json"
    assert load_partition_registry(missing) is None


def test_route_partition_by_keywords() -> None:
    registry = _registry()
    assert registry.route("糖尿病怎么诊断？") == "diagnosis"
    assert registry.route("阿司匹林有哪些用药注意事项？") == "medication"
    assert registry.route("阿司匹林禁忌是什么？") == "medication"
    assert registry.route("这个CT设备按新分类目录怎么注册？") == "device"
    assert registry.route("今天天气怎么样？") is None


def test_route_priority_is_stable_for_mixed_questions() -> None:
    registry = _registry()
    # 同时命中“影像/设备”(device) 与“诊断”(diagnosis) 时固定路由到 device
    assert registry.route("影像设备做诊断时如何操作？") == "device"
    assert registry.route("高血压患者用药需注意什么？") == "medication"


def test_labels_and_source_type() -> None:
    registry = _registry()
    assert registry.label("diagnosis") == "诊疗问题"
    assert registry.label("device") == "设备问题"
    assert registry.label("medication") == "用药问题"
    assert registry.label(None) == "相关问题"
    assert registry.label("unknown") == "相关问题"
    assert registry.source_type("diagnosis") == "clinical_guideline"
    assert registry.source_type("device") == "medical_device"
    assert registry.source_type("medication") == "medication_catalog"


def test_partition_metadata_template() -> None:
    registry = _registry()
    diagnosis = registry.partition("diagnosis")
    assert diagnosis is not None
    assert diagnosis.metadata_template["partition_id"] == "diagnosis"
    assert diagnosis.metadata_template["source_type"] == "clinical_guideline"
