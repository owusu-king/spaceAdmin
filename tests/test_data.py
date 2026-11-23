"""Data processing: normalization, validation, building resolution and notice parsing."""
import pandas as pd
import pytest

from app.data.buildings import BuildingResolver, floor_plan_files
from app.data.loader import DataValidationError, class_group, normalize_spaces, role_group
from app.data.notices import parse_date, parse_notice, split_messages


def test_dataset_shapes_and_quality(system):
    ds = system.ds
    assert len(ds.spaces) == 22408 and len(ds.people) == 16161
    assert ds.report["people_location_unmatched"] == 0
    assert ds.spaces["space_id"].is_unique            # duplicate room keys get distinct ids
    assert ds.report["duplicate_room_keys"] > 0


def test_original_identifiers_preserved(system):
    rooms = set(system.ds.spaces["room"])
    assert "000" in rooms and "001A" in rooms           # leading zeros and suffixes kept
    assert set(system.ds.spaces["floor"]) >= {"B", "01", "B2"}


@pytest.mark.parametrize("cls,group", [("250 - Research/Nonclass Laboratory", "Research Lab"),
                                       ("310 - Office", "Office"), ("Y03 - Shaft", "Building Service"),
                                       ("920 - Sleep/Study With Toilet or Bath", "Residential"), (None, "Unknown")])
def test_class_group_mapping(cls, group):
    assert class_group(cls) == group


def test_role_group_defaults_to_staff():
    assert role_group("Tenured/Tenure Track Faculty") == "faculty"
    assert role_group("Management - Manager") == "staff"


def test_missing_columns_rejected():
    with pytest.raises(DataValidationError):
        normalize_spaces(pd.DataFrame({"Building": ["X"]}))


def test_building_resolution_and_ambiguity():
    r = BuildingResolver(["Brockman Hall", "Brockman Hall for Opera", "Geology"])
    assert r.resolve("KWG").export_name == "Geology"
    assert r.resolve("MDA Biology").code == "ABL"
    assert r.resolve("Brockman").status == "ambiguous"     # never auto-resolved
    assert r.resolve("Nowhere Hall").status == "not_found"


def test_floor_plan_index_keeps_floor_labels():
    plans = floor_plan_files()
    assert len(plans) == 65
    assert any(p["code"] == "BRC" and p["floor"] == "02" and p["snapshot"] == "Nov 2026" for p in plans)


def test_date_formats():
    assert parse_date("October 27, 2026 at 8:27:27 AM CDT").hour == 8
    assert parse_date("Thu, Feb 26, 2026 at 1:27 PM").day == 26
    assert parse_date("Oct 22, 2026, at 8:23 AM").minute == 23
    assert parse_date("not a date") is None


def test_tririga_notice_fields_and_exclusions():
    text = ("The approved outage duration will be from 11/19/2026 07:00:00 to 11/19/2026 09:00:00.\n"
            "The outage will impact the following building(s): \nRalph S. O'Connor Building for Engineering and Science\n"
            "Following Systems Impacted: \nBuilding Exhaust\nShops Involved: \nZone 4\nOutage Request Summary\n"
            "Outage Request ID: SR-1119085\nScope of Work (What you can expect):\nThis is a reduction of exhaust flow. "
            "Chemical fume hoods and snorkels will not be affected.\nAffected Lab or Other Equipment:\nlab exhaust\n")
    n = parse_notice(text, BuildingResolver(), "Notification of Planned Outage: SR-1119085 - Scheduled Outage")
    assert n.sr == "SR-1119085" and n.kind == "planned_notification" and n.priority == "Scheduled"
    assert n.buildings == ["O'Connor Engineering and Science"] and n.systems == ["Building Exhaust"]
    assert n.duration_hours == 2.0
    assert any("fume hoods" in e for e in n.exclusions)


def test_quoted_history_not_counted_but_recoverable():
    text = ("From: A <a@rice.edu>\nSubject: Re: Elevator\nDate: October 27, 2026 at 8:35:32 AM CDT\nTo: b\n"
            "It is back in service.\nOn Tue, Oct 20, 2026 at 12:45 PM Juan Sanchez <cjds@rice.edu> wrote:\n"
            "The elevator is out of order.\n")
    assert len(split_messages(text)) == 1                       # repeated quotation is not a new event
    msgs = split_messages(text, include_quoted=True)
    assert len(msgs) == 2 and msgs[1].quoted and msgs[1].date.day == 20


def test_corpus_deduplicates_outage_requests(system):
    hist = system.corpus.history
    assert len(hist) == len({h["sr"] for h in hist}) == 40
