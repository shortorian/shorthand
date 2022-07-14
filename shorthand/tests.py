import pandas as pd
import shorthand as shnd
from bibtexparser.bparser import BibTexParser


def test_parsed_manual_annotation_has_62_strings_rows():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (len(parsed.strings) == 62)


def test_parsed_manual_annotation_string_id_58_startswith():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    string58 = parsed.strings.loc[58, 'string']

    assert (string58.startswith('This is stuff shorthand ignores'))


def test_parsed_manual_annotation_has_105_links_rows():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (len(parsed.links) == 105)


def test_parsed_manual_annotation_node_type_6_is_work():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (parsed.id_lookup('node_types', 'work', 'node_type') == 6)


def test_parsed_manual_annotation_has_18_strings_of_node_type_6():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (len(parsed.strings.query('node_type_id == 6')) == 18)


def test_parsed_manual_annotation_resolve_links_has_no_nans():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    assert (~parsed.resolve_links().isna()).any().any()


def test_manual_annotation_wrk_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    synthesized = parsed.synthesize_shorthand_entries('wrk', fill_spaces=True)

    check = pd.Series([
        'asmith_bwu__1999__s_bams__101__803__xxx',
        'asmith_bwu__1998__s_bams__100__42__yyy',
        'bjones__1975__s_jats__90__1__!',
        'bwu__1989__t_long|title__!__80__!',
        'Some|Author__1989__t_A|Title|With|\\#__!__!__!',
        'asmith_bwu__2008__s_bams__110__1__zzz'
    ])

    assert (check == synthesized).all()


def test_manual_annotation_note_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/manual_annotation.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=2,
        comment_char='#'
    )

    synthesized = parsed.synthesize_shorthand_entries(entry_node_type='note')

    check = pd.Series([
        'this is an article I made up for testing',
        'here\'s a note with an escaped\\__item separator and some "quotation marks"'
    ])

    assert (check == synthesized).all()


def test_single_column_wrk_synthesis():

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_entry_syntax.csv",
        link_syntax="shorthand/resources/default_link_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_text(
        'shorthand/test_data/single_column.shnd',
        item_separator='__',
        default_entry_prefix='wrk',
        space_char='|',
        na_string_values='!',
        na_node_type='missing',
        skiprows=0,
        comment_char='#',
        drop_na=False
    )

    synthesized = parsed.synthesize_shorthand_entries('wrk', fill_spaces=True)

    check = pd.Series([
        'asmith_bwu__1999__s_bams__101__803__xxx',
        'asmith_bwu__1998__s_bams__100__42__yyy',
        'bjones__1975__s_jats__90__1__!',
        'bwu__1989__t_long|title__!__80__!',
        'Some|Author__1989__t_A|Title|With|\\#__!__!__!',
        'asmith_bwu__2008__s_bams__110__1__zzz'
    ])

    assert (check == synthesized).all()


def test_parsed_bibtex_items_resolve_links_has_no_nans():

    bibtex_parser = BibTexParser(common_strings=True)
    bibtex_test_data_fname = "shorthand/test_data/bibtex_test_data_short.bib"
    with open(bibtex_test_data_fname, encoding='utf8') as f:
        bibdatabase = bibtex_parser.parse_file(f)

    data = pd.DataFrame(bibdatabase.entries)

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_bibtex_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_items(
        data.iloc[:4],
        space_char='|',
        na_string_values='!',
        na_node_type='missing'
    )

    assert (~parsed.resolve_links().isna()).any().any()


def test_bibtex_identifier_parsing():

    bibtex_parser = BibTexParser(common_strings=True)
    with open("shorthand/test_data/bibtex_test_data_short.bib", encoding='utf8') as f:
        bibdatabase = bibtex_parser.parse_file(f)

    data = pd.DataFrame(bibdatabase.entries)

    s = shnd.Shorthand(
        entry_syntax="shorthand/resources/default_bibtex_syntax.csv",
        syntax_case_sensitive=False
    )

    parsed = s.parse_items(
        data,
        space_char='|',
        na_string_values='!',
        na_node_type='missing'
    )

    parsed_identifiers = parsed.strings.query('node_type_id == 8')

    check = pd.Series([
        '10.1038/194638b0',
        '10.1175/1520-0493(1962)090<0311:OTOKEB>2.0.CO;2',
        '10.3402/tellusa.v14i3.9551',
        '10.1175/1520-0477-43.9.451',
        '10.3402/tellusa.v14i4.9569',
        '10.1007/BF02317953',
        '10.1007/BF02247180',
        '10.1029/JZ068i011p03345',
        '10.1029/JZ068i009p02375',
    ])

    assert (check == parsed_identifiers['string'].array).all()
