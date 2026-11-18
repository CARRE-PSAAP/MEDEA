"""Material and MaterialGroup: user-facing behaviour.

Tests are organised around how the class is used, not around lines of code:

* a non-transport user builds it from plain dicts and gets clear errors,
* data is stored safely (inputs copied, outputs read-only),
* a methods developer slices data at any level and converts to MFEM,
* the attribute map can be swapped without touching the data.
"""

import warnings

import numpy as np
import pytest

from medea._backend import mfem
from medea.materials import Material, MaterialGroup, MaterialProperty

# 2 groups, P1 scatter (G, G, L); numbers chosen so every entry is distinct.
SCATTER_A = np.arange(8.0).reshape(2, 2, 2)
SCATTER_B = 10 + SCATTER_A


@pytest.fixture
def mats():
    return MaterialGroup.from_dict(
        {
            "fuel": {"total": [1.0, 2.0], "scatter": SCATTER_A, "temp": 300.0},
            "mod": {"total": [3.0, 4.0], "scatter": SCATTER_B, "temp": 400.0},
        }
    )


def eval_on(coef, mesh, elem):
    return coef.Eval(mesh.GetElementTransformation(elem), mfem.IntegrationPoint())


def strip_mesh(attrs):
    """Mesh with one element per entry of ``attrs``, element i has attrs[i]."""
    mesh = mfem.Mesh(len(attrs), 1.0)
    for e, a in enumerate(attrs):
        mesh.SetAttribute(e, a)
    mesh.SetAttributes()
    return mesh


# -- construction: the plain-dict path non-transport users take ----------------


def test_default_attributes_follow_dict_order(mats):
    assert mats.attributes == {1: "fuel", 2: "mod"}
    assert mats.names == ("fuel", "mod")


def test_properties_are_discoverable(mats):
    assert list(mats) == ["total", "scatter", "temp"]
    assert len(mats) == 3


def test_accepts_lists_scalars_and_arrays_of_any_shape(mats):
    assert mats["temp"].shape == ()
    assert mats["total"].shape == (2,)
    assert mats["scatter"].shape == (2, 2, 2)
    assert mats["total"].array.dtype == np.float64


def test_integer_input_becomes_float():
    m = MaterialGroup.from_dict({"a": {"x": [1, 2]}})
    assert m["x"].array.dtype == np.float64


def test_property_names_are_not_special():
    m = MaterialGroup.from_dict({"a": {"stopping_power": 1.0, "weird name": 2.0}})
    assert m["weird name"]["a"] == 2.0


@pytest.mark.parametrize(
    "materials, match",
    [
        ({}, "non-empty"),
        ({"a": {"x": 1.0}, "b": {"y": 1.0}}, "same properties"),
        ({"a": {"x": 1.0}, "b": {"x": 1.0, "y": 2.0}}, "same properties"),
        ({"a": {"x": [1.0, 2.0]}, "b": {"x": [1.0, 2.0, 3.0]}}, "shape"),
        ({"a": {"x": 1.0}, "b": {"x": [1.0]}}, "shape"),
    ],
)
def test_inconsistent_materials_rejected_with_useful_message(materials, match):
    with pytest.raises(ValueError, match=match):
        MaterialGroup.from_dict(materials)


def test_error_names_the_offending_material_and_property():
    with pytest.raises(ValueError) as e:
        MaterialGroup.from_dict({"good": {"sig": [1.0]}, "bad": {"sig": [1.0, 2.0]}})
    assert "bad" in str(e.value) and "sig" in str(e.value)


# -- data safety ----------------------------------------------------------------


def test_mutating_input_after_construction_does_not_change_data():
    src = np.array([1.0, 2.0])
    m = MaterialGroup.from_dict({"a": {"x": src}})
    src[0] = 99.0
    assert m["x"]["a"][0] == 1.0


def test_stored_data_is_read_only(mats):
    with pytest.raises(ValueError):
        mats["total"].array[0, 0] = 5.0
    with pytest.raises(ValueError):
        mats["total"]["fuel"][0] = 5.0
    with pytest.raises(ValueError):
        mats["total"][0].array[0] = 5.0


# -- attribute map ----------------------------------------------------------------


def test_sparse_attributes_and_material_on_several_attributes():
    m = MaterialGroup.from_dict(
        {"fuel": {"x": 1.0}, "mod": {"x": 2.0}},
        attributes={1: "fuel", 3: "mod", 7: "fuel"},
    )
    assert m.attributes == {1: "fuel", 3: "mod", 7: "fuel"}
    mesh = strip_mesh([7, 3, 1])
    c = m["x"].to_PWConst()
    assert [eval_on(c, mesh, e) for e in range(3)] == [1.0, 2.0, 1.0]


def test_numpy_integer_attribute_keys_accepted():
    m = MaterialGroup.from_dict({"a": {"x": 1.0}}, attributes={np.int64(2): "a"})
    assert m.attributes == {2: "a"}


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1"])
def test_invalid_attribute_keys_rejected(bad):
    with pytest.raises(ValueError, match="positive integers"):
        MaterialGroup.from_dict({"a": {"x": 1.0}}, attributes={bad: "a"})


def test_attribute_naming_unknown_material_rejected():
    with pytest.raises(ValueError, match="unknown materials.*typo"):
        MaterialGroup.from_dict({"a": {"x": 1.0}}, attributes={1: "typo"})


def test_empty_attribute_map_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        MaterialGroup.from_dict({"a": {"x": 1.0}}, attributes={})

def test_no_warning_when_every_material_is_assigned(mats):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MaterialGroup.from_dict(
            {"a": {"x": 1.0}, "b": {"x": 2.0}}, attributes={5: "a", 6: "b"}
        )

def test_remap_without_copying_or_mutating(mats):
    swapped = mats.remap({1: "mod", 2: "fuel"})
    assert swapped.attributes == {1: "mod", 2: "fuel"}
    assert mats.attributes == {1: "fuel", 2: "mod"}  # original untouched
    assert np.shares_memory(swapped["total"].array, mats["total"].array)

    mesh = strip_mesh([1, 2])
    old = mats["total"][0].to_PWConst()
    new = swapped["total"][0].to_PWConst()
    assert [eval_on(old, mesh, e) for e in (0, 1)] == [1.0, 3.0]
    assert [eval_on(new, mesh, e) for e in (0, 1)] == [3.0, 1.0]


def test_remap_validates_like_the_constructor(mats):
    with pytest.raises(ValueError, match="unknown materials"):
        mats.remap({1: "nope"})
    with pytest.raises(ValueError, match="positive integers"):
        mats.remap({0: "fuel"})


def test_attributes_property_is_a_copy(mats):
    mats.attributes[1] = "mod"
    assert mats.attributes[1] == "fuel"


def test_check_mesh_accepts_unused_mapped_attributes(mats):
    mats.check_mesh(strip_mesh([1, 1, 1]))  # attribute 2 unused: fine


def test_check_mesh_reports_unmapped_mesh_attributes(mats):
    with pytest.raises(ValueError, match=r"\[3\]"):
        mats.check_mesh(strip_mesh([1, 3, 2]))


# -- indexing: the "any level" access methods developers rely on -------------------


def test_view_is_dict_like_over_material_names(mats):
    v = mats["total"]
    assert list(v) == ["fuel", "mod"]
    assert list(v.keys()) == ["fuel", "mod"]
    assert len(v) == 2
    assert "fuel" in v and "water" not in v
    np.testing.assert_array_equal(v["mod"], [3.0, 4.0])
    assert {k: x.tolist() for k, x in v.items()} == {
        "fuel": [1.0, 2.0],
        "mod": [3.0, 4.0],
    }


def test_view_converts_to_a_plain_dict(mats):
    d = dict(mats["total"][0])
    assert d == {"fuel": 1.0, "mod": 3.0}


def test_integer_index_selects_leading_non_material_axis(mats):
    v = mats["total"][1]
    assert v.shape == ()
    assert dict(v) == {"fuel": 2.0, "mod": 4.0}


def test_chained_and_tuple_indexing_agree(mats):
    s = mats["scatter"]
    np.testing.assert_array_equal(s[1][0][1].array, s[1, 0, 1].array)
    np.testing.assert_array_equal(
        s[1, 0, 1].array, [SCATTER_A[1, 0, 1], SCATTER_B[1, 0, 1]]
    )


def test_slice_and_ellipsis_indexing_keep_material_axis_first(mats):
    s = mats["scatter"]
    assert s[:, 0].shape == (2, 2)
    assert s[..., 1].shape == (2, 2)
    np.testing.assert_array_equal(s[..., 1]["mod"], SCATTER_B[..., 1])
    np.testing.assert_array_equal(s[0, :, 1]["fuel"], SCATTER_A[0, :, 1])


def test_indexing_does_not_copy(mats):
    assert np.shares_memory(mats["scatter"][0, :, 1].array, mats["scatter"].array)
    assert np.shares_memory(mats["scatter"]["fuel"], mats["scatter"].array)


def test_unknown_names_give_helpful_keyerrors(mats):
    with pytest.raises(KeyError, match="unknown property 'tota'.*total"):
        mats["tota"]
    with pytest.raises(KeyError, match="unknown material 'water'.*fuel"):
        mats["total"]["water"]


def test_over_indexing_raises(mats):
    with pytest.raises(IndexError):
        mats["total"][0, 0]
    with pytest.raises(IndexError):
        mats["temp"][0]


# -- MFEM conversion at every level --------------------------------------------------


def test_scalar_property_converts_to_single_coefficient(mats):
    c = mats["temp"].to_PWConst()
    assert isinstance(c, mfem.PWConstCoefficient)
    mesh = strip_mesh([1, 2])
    assert [eval_on(c, mesh, e) for e in (0, 1)] == [300.0, 400.0]


def test_vector_property_converts_to_object_array_of_coefficients(mats):
    c = mats["total"].to_PWConst()
    assert c.shape == (2,) and c.dtype == object
    mesh = strip_mesh([1, 2])
    assert [eval_on(c[g], mesh, 0) for g in range(2)] == [1.0, 2.0]
    assert [eval_on(c[g], mesh, 1) for g in range(2)] == [3.0, 4.0]


def test_scalar_after_indexing_converts_to_single_coefficient(mats):
    c = mats["total"][1].to_PWConst()
    assert isinstance(c, mfem.PWConstCoefficient)
    assert eval_on(c, strip_mesh([2]), 0) == 4.0


def test_full_tensor_conversion_matches_data_at_every_entry(mats):
    """Whole-array conversion: coefficient [g, h, l] on attr a is that material's value."""
    c = mats["scatter"].to_PWConst()
    assert c.shape == (2, 2, 2)
    mesh = strip_mesh([1, 2])
    for idx in np.ndindex(2, 2, 2):
        assert eval_on(c[idx], mesh, 0) == SCATTER_A[idx]
        assert eval_on(c[idx], mesh, 1) == SCATTER_B[idx]


def test_converting_a_slice_equals_slicing_the_conversion(mats):
    mesh = strip_mesh([1, 2])
    sub = mats["scatter"][:, 1, 0].to_PWConst()
    full = mats["scatter"].to_PWConst()
    for g in range(2):
        for a in range(2):
            assert eval_on(sub[g], mesh, a) == eval_on(full[g, 1, 0], mesh, a)


def test_unmapped_attribute_gets_zero_inside_mapped_range():
    m = MaterialGroup.from_dict(
        {"a": {"x": 5.0}, "b": {"x": 6.0}}, attributes={1: "a", 3: "b"}
    )
    c = m["x"].to_PWConst()
    mesh = strip_mesh([1, 2, 3])  # attribute 2 has no material
    assert [eval_on(c, mesh, e) for e in range(3)] == [5.0, 0.0, 6.0]


def test_conversion_is_a_snapshot_of_the_attribute_map(mats):
    """Coefficients made before remap keep the old mapping."""
    c = mats["total"][0].to_PWConst()
    mats.remap({1: "mod", 2: "fuel"})
    assert eval_on(c, strip_mesh([1]), 0) == 1.0


def test_single_material_problem():
    m = MaterialGroup.from_dict({"only": {"sig": [2.0, 3.0]}})
    c = m["sig"].to_PWConst()
    mesh = strip_mesh([1, 1])
    assert eval_on(c[1], mesh, 1) == 3.0


def test_coefficients_work_in_a_real_form():
    """End to end: material-dependent mass matrix integrates to sum(sigma_a * area_a)."""
    m = MaterialGroup.from_dict({"a": {"s": 2.0}, "b": {"s": 5.0}})
    mesh = mfem.Mesh(4, 1.0)  # [0,1], 4 elements of length 0.25
    for e in range(4):
        mesh.SetAttribute(e, 1 if e < 2 else 2)
    mesh.SetAttributes()
    m.check_mesh(mesh)
    fes = mfem.FiniteElementSpace(mesh, mfem.L2_FECollection(0, 1), 1)
    one = mfem.GridFunction(fes)
    one.Assign(1.0)
    form = mfem.LinearForm(fes)
    form.AddDomainIntegrator(mfem.DomainLFIntegrator(m["s"].to_PWConst()))
    form.Assemble()
    assert form.Sum() == pytest.approx(2.0 * 0.5 + 5.0 * 0.5)


# -- Material: a single object ---------------------------------------------------------


def test_material_stores_name_and_read_only_float_data():
    src = np.array([1, 2])
    m = Material("fuel", {"total": src, "temp": 300})
    src[0] = 99
    assert m.name == "fuel"
    assert m["total"].dtype == np.float64 and m["total"][0] == 1.0
    assert m.shapes == {"total": (2,), "temp": ()}
    assert list(m) == ["total", "temp"] and "temp" in m and len(m) == 2
    with pytest.raises(ValueError):
        m["total"][0] = 5.0


@pytest.mark.parametrize("name", ["", None, 3])
def test_material_needs_a_nonempty_string_name(name):
    with pytest.raises(ValueError, match="name"):
        Material(name, {"x": 1.0})


def test_material_needs_properties():
    with pytest.raises(ValueError, match="at least one property"):
        Material("a", {})


def test_material_unknown_property_names_the_material():
    with pytest.raises(KeyError, match="'a'.*'y'.*x"):
        Material("a", {"x": 1.0})["y"]


def test_replace_returns_a_new_material_and_leaves_the_original():
    a = Material("a", {"x": 1.0, "y": 2.0})
    b = a.replace(x=5.0, z=7.0)
    assert b.name == "a" and b["x"] == 5.0 and b["y"] == 2.0 and b["z"] == 7.0
    assert a["x"] == 1.0 and "z" not in a


# -- MaterialGroup built from Material objects ---------------------------------------------


def test_group_takes_materials_as_positional_arguments():
    a, b = Material("a", {"x": 1.0}), Material("b", {"x": 2.0})
    g = MaterialGroup(a, b)
    assert g.names == ("a", "b") and g.materials == (a, b)
    assert g.attributes == {1: "a", 2: "b"}
    assert g.material("b") is b
    assert g.properties == ("x",)


def test_attributes_accept_material_objects_and_names():
    a, b = Material("a", {"x": 1.0}), Material("b", {"x": 2.0})
    g = MaterialGroup(a, b, attributes={1: a, 2: "b", 3: b})
    assert g.attributes == {1: "a", 2: "b", 3: "b"}


def test_attribute_material_from_another_group_is_rejected():
    a = Material("a", {"x": 1.0})
    impostor = Material("a", {"x": 1.0})
    with pytest.raises(ValueError, match="not part of this group"):
        MaterialGroup(a, attributes={1: impostor})
    with pytest.raises(ValueError, match="not part of this group"):
        MaterialGroup(a, attributes={1: Material("zzz", {"x": 1.0})})


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate.*'a'"):
        MaterialGroup(Material("a", {"x": 1.0}), Material("a", {"x": 2.0}))


def test_group_rejects_non_materials_and_empty_input():
    with pytest.raises(TypeError, match="from_dict"):
        MaterialGroup({"a": {"x": 1.0}})
    with pytest.raises(ValueError, match="at least one material"):
        MaterialGroup()

def test_group_material_lookup_error_lists_materials(mats):
    with pytest.raises(KeyError, match="unknown material 'water'.*fuel"):
        mats.material("water")


def test_require_checks_properties(mats):
    mats.require("total", "scatter")
    with pytest.raises(ValueError, match=r"missing properties \['nope'\].*total"):
        mats.require("total", "nope")


def test_num_groups_defaults_to_one_and_survives_remap(mats):
    assert mats.num_groups == 1
    two = MaterialGroup.from_dict({"a": {"x": [1.0, 2.0]}}, num_groups=2)
    assert two.num_groups == 2
    assert two.remap({1: "a"}).num_groups == 2
    assert "num_groups=2" in repr(two)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_num_groups_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="num_groups"):
        MaterialGroup.from_dict({"a": {"x": 1.0}}, num_groups=bad)


def test_require_checks_shapes_against_num_groups():
    mats = MaterialGroup.from_dict(
        {"a": {"total": [1.0, 2.0], "scatter": SCATTER_A[:, :, 0], "temp": 300.0}},
        num_groups=2,
    )
    G = mats.num_groups
    mats.require("temp", total=(G,), scatter=(G, G))
    mats.require(total=G, temp=())  # ints are one-dimensional shapes
    with pytest.raises(ValueError, match=r"'scatter'.*\(2, 2\).*expected \(2, 2, 2\)"):
        mats.require(scatter=(G, G, 2))


def test_require_scalar_does_not_match_one_group_shape():
    mats = MaterialGroup.from_dict({"a": {"total": 1.0}})
    with pytest.raises(ValueError, match=r"'total'.*\(\).*expected \(1,\)"):
        mats.require(total=(mats.num_groups,))


def test_require_reports_missing_keyword_properties():
    mats = MaterialGroup.from_dict({"a": {"total": [1.0]}})
    with pytest.raises(ValueError, match=r"missing properties \['nope'\]"):
        mats.require(nope=(1,))


def test_group_does_not_alias_its_material_data(mats):
    assert isinstance(mats["total"], MaterialProperty)
    assert not np.shares_memory(mats["total"].array, mats.material("fuel")["total"])


# -- coefficients are cached at construction ----------------------------------------------


def test_coefficients_are_built_once_and_reused(mats):
    first = mats["scatter"].to_PWConst()
    second = mats["scatter"].to_PWConst()
    assert first is not second  # fresh container, so callers can't corrupt the cache
    assert all(x is y for x, y in zip(first.flat, second.flat))
    assert mats["total"].to_PWConst()[1] is mats["total"][1].to_PWConst()
    assert mats["temp"].to_PWConst() is mats["temp"].to_PWConst()


def test_mutating_returned_container_does_not_poison_cache(mats):
    c = mats["total"].to_PWConst()
    c[0] = None
    assert mats["total"].to_PWConst()[0] is not None


def test_remap_rebuilds_coefficients_and_leaves_the_original_cache(mats):
    swapped = mats.remap({1: "mod", 2: "fuel"})
    assert swapped["temp"].to_PWConst() is not mats["temp"].to_PWConst()
    mesh = strip_mesh([1])
    assert eval_on(mats["temp"].to_PWConst(), mesh, 0) == 300.0
    assert eval_on(swapped["temp"].to_PWConst(), mesh, 0) == 400.0